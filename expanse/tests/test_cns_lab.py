"""Tests for the Connectome Lab harness (``cns_lab/``): pre-registration, verdicts, tasks, analysis.

* lock      -> a lock records the file hash; an edited hypothesis is refused by
               ``load_locked`` and cannot be re-locked;
* verdict   -> SUPPORTED / NOT SUPPORTED / INCONCLUSIVE from synthetic run metrics,
               the delta sign follows ``direction``, relative thresholds use the
               control mean, and an issued verdict is never rewritten;
* run keys  -> arms shared by two hypotheses with one protocol get one run;
* tasks     -> deterministic, held-out prompts never in the train rows, answers
               derivable from the prompt;
* analysis  -> modularity and drift report on a synthetic core file.

Run: ``python -m pytest tests/test_cns_lab.py -q``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cns_lab"))
sys.path.insert(0, str(ROOT / "tests"))

import connectome_full as cf  # noqa: E402
import connectome_lab as cl  # noqa: E402
import prereg  # noqa: E402
import temporal_tasks as tt  # noqa: E402

PROTOCOL = {"steps": 10, "batch": 2, "eval_steps": [5, 10], "base_config": {"edge_param": "dale_softplus"}}


def hyp(hid="H-T-001", rule=None, direction="lower_is_better", seeds=(1, 2, 3)):
    return {"id": hid, "claim": "real beats shuffled", "protocol": PROTOCOL,
            "arms": {"real": {"topology": "real"}, "shuffled": {"topology": "shuffled"}},
            "control_arm": "shuffled", "treatment_arm": "real", "seeds": list(seeds),
            "primary_metric": "val_nats_per_char_pooled", "direction": direction,
            "success_rule": rule or {"all_same_sign": True, "min_mean_delta_rel": 0.01}}


def write_hyp(d: Path, h) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{h['id']}.json").write_text(json.dumps(h, indent=1), encoding="utf-8")


def fake_run(runs: Path, h, arm, seed, value, status="done", steps=None):
    d = prereg.run_dir(h, arm, seed, runs)
    d.mkdir(parents=True, exist_ok=True)
    m = {"status": status, "steps_done": h["protocol"]["steps"] if steps is None else steps,
         "evals": [{"step": 5, h["primary_metric"]: 9.0}, {"step": 10, h["primary_metric"]: value}]}
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")


def test_lock_and_tamper(tmp_path):
    hd = tmp_path / "hyp"
    h = hyp()
    write_hyp(hd, h)
    with pytest.raises(prereg.PreregError):
        prereg.load_locked(h["id"], hd)  # not locked yet
    rec = prereg.lock(h["id"], hd)
    assert rec["sha256"] == prereg.sha256_bytes((hd / "H-T-001.json").read_bytes())
    assert prereg.lock(h["id"], hd)["sha256"] == rec["sha256"]  # idempotent
    assert prereg.load_locked(h["id"], hd)["_lock"]["sha256"] == rec["sha256"]
    h2 = dict(h, seeds=[1, 2])
    write_hyp(hd, h2)
    with pytest.raises(prereg.PreregError):
        prereg.load_locked(h["id"], hd)
    with pytest.raises(prereg.PreregError):
        prereg.lock(h["id"], hd)


def test_validation():
    prereg.validate(hyp())
    for bad in (dict(hyp(), control_arm="nope"), dict(hyp(), seeds=[1]), dict(hyp(), direction="up"),
                dict(hyp(), success_rule={"all_same_sign": True}),
                dict(hyp(), arms={"real": {"topology": "bogus"}, "shuffled": {}})):
        with pytest.raises((prereg.PreregError, ValueError)):
            prereg.validate(bad)
    prereg.validate({"id": "X", "claim": "explore", "exploratory": True, "protocol": PROTOCOL,
                     "arms": {"tanh": {"activation": "tanh"}}, "seeds": [1]})


def test_three_verdicts(tmp_path):
    hd, runs, vd = tmp_path / "hyp", tmp_path / "runs", tmp_path / "verdicts"
    h = hyp()
    write_hyp(hd, h)
    prereg.lock(h["id"], hd)
    hl = prereg.load_locked(h["id"], hd)
    # nothing ran: INCONCLUSIVE, and it is not written
    v = prereg.verdict(h["id"], hd, runs, vd)
    assert v["verdict"] == "INCONCLUSIVE" and not (vd / "H-T-001.verdict.json").exists()
    # real better on every seed by > 1% of the control mean: SUPPORTED
    for s, (c, t) in zip((1, 2, 3), ((1.00, 0.97), (1.02, 0.99), (0.98, 0.96))):
        fake_run(runs, hl, "shuffled", s, c)
        fake_run(runs, hl, "real", s, t)
    res = prereg.evaluate_rule(hl, runs)
    assert res["verdict"] == "SUPPORTED" and res["all_same_sign"] and res["mean_delta"] > res["threshold"]
    assert res["threshold"] == pytest.approx(0.01 * (1.00 + 1.02 + 0.98) / 3)
    # one seed against: NOT SUPPORTED
    fake_run(runs, hl, "real", 3, 0.99)
    assert prereg.evaluate_rule(hl, runs)["verdict"] == "NOT SUPPORTED"
    # a failed / short / NaN run: INCONCLUSIVE
    fake_run(runs, hl, "real", 3, 0.96, status="failed")
    assert prereg.evaluate_rule(hl, runs)["verdict"] == "INCONCLUSIVE"
    fake_run(runs, hl, "real", 3, 0.96, steps=5)
    assert prereg.evaluate_rule(hl, runs)["verdict"] == "INCONCLUSIVE"
    fake_run(runs, hl, "real", 3, float("nan"))
    assert prereg.evaluate_rule(hl, runs)["verdict"] == "INCONCLUSIVE"
    # issued once, then immutable even if the metrics change
    fake_run(runs, hl, "real", 3, 0.96)
    first = prereg.verdict(h["id"], hd, runs, vd)
    assert first["verdict"] == "SUPPORTED" and (vd / "H-T-001.verdict.json").exists()
    fake_run(runs, hl, "real", 3, 1.5)
    again = prereg.verdict(h["id"], hd, runs, vd)
    assert again["verdict"] == "SUPPORTED" and "immutable" in again["_note"]


def test_direction_and_absolute_rule(tmp_path):
    hd, runs = tmp_path / "hyp", tmp_path / "runs"
    h = hyp("H-T-002", rule={"all_same_sign": True, "min_mean_delta": 0.02}, direction="higher_is_better", seeds=(1, 2))
    write_hyp(hd, h)
    prereg.lock(h["id"], hd)
    hl = prereg.load_locked(h["id"], hd)
    for s in (1, 2):
        fake_run(runs, hl, "shuffled", s, 0.40)
        fake_run(runs, hl, "real", s, 0.43)
    r = prereg.evaluate_rule(hl, runs)
    assert r["verdict"] == "SUPPORTED" and r["mean_delta"] == pytest.approx(0.03)
    fake_run(runs, hl, "real", 2, 0.41)
    assert prereg.evaluate_rule(hl, runs)["verdict"] == "NOT SUPPORTED"  # mean 0.02 - eps < 0.02


def test_shared_arms_share_runs():
    a, b = hyp("H-A"), hyp("H-B")
    b["arms"] = {"real": {"topology": "real"}, "none": {"topology": "none"}}
    b["control_arm"] = "none"
    assert prereg.run_dir(a, "real", 1) == prereg.run_dir(b, "real", 1)
    assert prereg.run_dir(a, "real", 1) != prereg.run_dir(a, "real", 2)
    assert prereg.run_dir(a, "real", 1) != prereg.run_dir(a, "shuffled", 1)
    c = hyp("H-C")
    c["protocol"] = dict(PROTOCOL, steps=11)
    assert prereg.run_dir(a, "real", 1) != prereg.run_dir(c, "real", 1)


def test_temporal_tasks():
    s1, s2 = tt.split_rows(200, 40, seed=7), tt.split_rows(200, 40, seed=7)
    assert s1 == s2
    assert len(s1["train"]) == 200 and len(s1["heldout"]) == 40
    assert not {r["user"] for r in s1["train"]} & {r["user"] for r in s1["heldout"]}
    assert {r["task"] for r in s1["heldout"]} == set(tt.FAMILIES)
    for r in s1["train"] + s1["heldout"]:
        u, a = r["user"], r["assistant"]
        if r["task"] == "recall":
            key = u.rsplit("code for ", 1)[1].rstrip("?")
            assert f"{key} {a}," in u + "," or f"{key} {a}." in u
        elif r["task"] == "count":
            target = u.split("'")[1]
            assert u.split("? ", 1)[1].split().count(target) == int(a)
        elif r["task"] == "first":
            assert u.split(": ", 1)[1].split(",")[0].strip() == a
        elif r["task"] == "pattern":
            seq = u.split(": ", 1)[1].split() + a.split()
            period = next(p for p in (2, 3, 4) if all(seq[i] == seq[i % p] for i in range(len(seq))))
            assert period in (2, 3, 4)
    assert tt.exact_match(" 3.", "3") and not tt.exact_match("4", "3")


def test_modularity_and_analysis(tmp_path, monkeypatch):
    import analyze
    # two cliques joined by one weak edge: high modularity under the true split, ~0 under a random one
    src = np.array([0, 1, 2, 0, 3, 4, 5, 3, 2])
    dst = np.array([1, 2, 0, 2, 4, 5, 3, 5, 3])
    w = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0.1])
    q_true = analyze.modularity(src, dst, w, np.array([0, 0, 0, 1, 1, 1]))
    q_bad = analyze.modularity(src, dst, w, np.array([0, 1, 0, 1, 0, 1]))
    assert q_true > 0.4 > q_bad

    from test_connectome_full import write_tiny_npz

    base = cf.build_full_graph(write_tiny_npz(tmp_path / "tiny.npz"))
    cfg = cl.LabConfig(edge_param="dale_softplus")
    g = cl.lab_graph(base, cfg, 0)
    core = cl.fresh_lab_core(16, g, cfg, seed=0)
    with torch.no_grad():
        core.edge_theta.add_(torch.randn(core.nnz) * 0.3)
    runs = tmp_path / "runs"
    monkeypatch.setattr(prereg, "RUNS", runs)
    monkeypatch.setattr(analyze, "AN", tmp_path / "an")
    (tmp_path / "an").mkdir()
    cl.save_core_file(runs / "k1" / "core.pt", core)
    rep = analyze.analyse_core("k1", {"arm": "real", "seed": 1, "evals": [{"gate_by_source": {"code": 0.1}}]})
    assert rep["sign_flips"] == 0 and rep["n_edges"] == core.nnz
    assert sum(gp["n"] for gp in rep["R_by_pathway"]) == core.nnz
    assert rep["R_quantiles"]["min"] > 0
    md = analyze.run_markdown(rep)
    assert "Edge drift" in md and "Most strengthened" in md
