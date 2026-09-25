"""Tests for the Connectome Lab harness (``cns_lab/``): pre-registration, verdicts, tasks, analysis.

* lock      -> a lock records the file hash; an edited hypothesis is refused by
               ``load_locked`` and cannot be re-locked;
* verdict   -> SUPPORTED / NOT SUPPORTED / INCONCLUSIVE from synthetic run metrics,
               the delta sign follows ``direction``, relative thresholds use the
               control mean, and an issued verdict is never rewritten;
* provenance-> a recorded run is used only if its run key, arm, seed (an exact int),
               config, lock (which must be recorded) and last eval step are the ones
               the hypothesis asks for, its ``untrainable`` flag is what the arm
               implies, and it was not an ``--unlocked`` run (runs shared with another
               hypothesis are checked against that hypothesis, which must be locked);
               malformed fields give INCONCLUSIVE, never an exception; verdict v2
               fields and input hashes (line endings normalised, as for the locks);
* run keys  -> arms shared by two hypotheses with one protocol get one run;
* tasks     -> deterministic, held-out prompts never in the train rows, answers
               derivable from the prompt; split v2 drops near-duplicates;
* run_lab   -> (tiny synthetic LM + lab core, no checkpoint) per-row final losses
               add up to the reported metrics, answer-only temporal scoring, core
               write diagnostics over every eval token (independent of the eval
               batch size), final ablations restore the core bit for bit,
               ``eps_edge`` reaches only the edge group, hash eval sampling, a
               never-locked hypothesis dict is recorded as unlocked;
* analysis  -> modularity and drift report on a synthetic core file; exploratory
               arms section with a noise floor from paired pre-registered runs.

Run: ``python -m pytest tests/test_cns_lab.py -q``
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cns_lab"))
sys.path.insert(0, str(ROOT / "tests"))

import connectome_full as cf  # noqa: E402
import connectome_lab as cl  # noqa: E402
import prereg  # noqa: E402
import temporal_tasks as tt  # noqa: E402

torch.set_num_threads(2)
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


def fake_run(runs: Path, h, arm, seed, value, /, status="done", steps=None, **fields):
    """A metrics.json as run_lab writes it for ``(h, arm, seed)``; ``fields`` override any key (even arm / seed)."""
    d = prereg.run_dir(h, arm, seed, runs)
    d.mkdir(parents=True, exist_ok=True)
    m = {"status": status, "steps_done": h["protocol"]["steps"] if steps is None else steps,
         "evals": [{"step": 5, h["primary_metric"]: 9.0}, {"step": 10, h["primary_metric"]: value}],
         "run_key": d.name, "hypothesis": h["id"], "arm": arm, "seed": int(seed),
         "config": prereg.arm_config(h, arm).to_dict(), "unlocked": False, "lock": h.get("_lock")}
    m.update(fields)
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    return d


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


def locked(tmp_path, h):
    hd = tmp_path / "hyp"
    write_hyp(hd, h)
    prereg.lock(h["id"], hd)
    return hd, prereg.load_locked(h["id"], hd)


def supported_runs(runs, hl):
    for s, (c, t) in zip((1, 2, 3), ((1.00, 0.97), (1.02, 0.99), (0.98, 0.96))):
        fake_run(runs, hl, "shuffled", s, c)
        fake_run(runs, hl, "real", s, t)


def test_verdict_v2_fields_and_inputs(tmp_path):
    hd, hl = locked(tmp_path, hyp())
    runs = tmp_path / "runs"
    supported_runs(runs, hl)
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "SUPPORTED" and res["verdict_schema_version"] == prereg.VERDICT_SCHEMA_VERSION == 2
    assert res["schema"] == prereg.VERDICT_SCHEMA  # issued v1 readers keep working
    assert res["all_favour_treatment"] is True and res["all_same_sign"] is True
    assert len(res["inputs"]) == 6 and res["inputs_hash"] == prereg.INPUTS_HASH == "sha256-lf-normalised"
    for rec in res["inputs"]:
        mp = prereg.run_dir(hl, rec["arm"], rec["seed"], runs) / "metrics.json"
        assert rec["run"] == mp.parent.name and rec["metrics_sha256"] == prereg.sha256_bytes(mp.read_bytes())
    assert {(r["role"], r["arm"]) for r in res["inputs"]} == {("control", "shuffled"), ("treatment", "real")}
    # input hashes use the lock convention (CRLF -> LF): a CRLF working copy of a metrics.json (what
    # Path.write_text gives on Windows) hashes like the LF blob git stores
    mp = prereg.run_dir(hl, "real", 1, runs) / "metrics.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    lf = json.dumps(m, indent=1).encode("utf-8")
    mp.write_bytes(lf)
    want = hashlib.sha256(lf).hexdigest()
    mp.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert b"\r\n" in mp.read_bytes() and hashlib.sha256(mp.read_bytes()).hexdigest() != want
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "SUPPORTED"
    assert [r["metrics_sha256"] for r in res["inputs"] if (r["arm"], r["seed"]) == ("real", 1)] == [want]
    # every delta negative: "all_same_sign" (= all favour the treatment) is False, as is its new name
    for s in (1, 2, 3):
        fake_run(runs, hl, "real", s, 1.2)
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "NOT SUPPORTED" and res["all_favour_treatment"] is False and res["all_same_sign"] is False
    # an INCONCLUSIVE verdict carries the version and the inputs it did read
    (prereg.run_dir(hl, "real", 3, runs) / "metrics.json").unlink()
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and res["verdict_schema_version"] == 2 and len(res["inputs"]) == 5


@pytest.mark.parametrize("tamper,expect", [
    ({"run_key": "0123456789abcdef"}, "run_key"),
    ({"arm": "shuffled"}, "recorded arm"),
    ({"seed": 7}, "recorded seed"),
    ({"seed": 3.9}, "recorded seed"),  # int() would truncate it to 3
    ({"seed": 3.0}, "recorded seed"),
    ({"seed": "3"}, "recorded seed"),
    ({"config": cl.LabConfig(edge_param="dale_softplus", microsteps=3).to_dict()}, "config"),
    ({"unlocked": True}, "unlocked"),
    ({"lock": {"id": "H-T-001", "sha256": "0" * 64}}, "lock sha"),
    ({"lock": None}, "no lock recorded"),  # e.g. run_lab.run called with a dict that was never locked
    ({"lock": {}}, "no lock recorded"),
    ({"lock": {"id": "H-T-001", "sha256": None}}, "no lock recorded"),
    ({"lock": "H-T-001"}, "no lock recorded"),
    ({"evals": [{"step": 5, "val_nats_per_char_pooled": 0.96}]}, "last eval"),
    ({"evals": [{"step": 10.5, "val_nats_per_char_pooled": 0.96}]}, "last eval"),
    ({"untrainable": True}, "untrainable"),  # a trainable arm cannot opt out of the step checks
    # malformed fields: a problem (-> INCONCLUSIVE), never an exception
    ({"steps_done": None}, "steps, protocol"),
    ({"steps_done": "x"}, "steps, protocol"),
    ({"evals": {"step": 10, "val_nats_per_char_pooled": 0.96}}, "last eval"),
    ({"evals": [5, 10]}, "last eval"),
    ({"evals": [{"step": 10, "val_nats_per_char_pooled": True}]}, "metric"),
    ({"evals": [{"step": 10, "val_nats_per_char_pooled": 10 ** 400}]}, "malformed metrics.json field"),
])
def test_verdict_rejects_runs_it_cannot_trace(tmp_path, tamper, expect):
    hd, hl = locked(tmp_path, hyp())
    runs = tmp_path / "runs"
    supported_runs(runs, hl)
    assert prereg.evaluate_rule(hl, runs, hd)["verdict"] == "SUPPORTED"
    fake_run(runs, hl, "real", 3, 0.96, **tamper)
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE", tamper
    assert any(p.startswith("real seed 3:") and expect in p for p in res["problems"]), res["problems"]
    assert len(res["problems"]) == 1, res["problems"]


def test_verdict_seed_is_compared_exactly(tmp_path):
    """True == 1 in Python: a recorded seed True is not seed 1."""
    hd, hl = locked(tmp_path, hyp())
    runs = tmp_path / "runs"
    supported_runs(runs, hl)
    fake_run(runs, hl, "real", 1, 0.97, seed=True)
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and res["problems"] == ["real seed 1: recorded seed True != 1"]


def test_verdict_untrainable_and_unreadable_runs(tmp_path):
    h = hyp()
    h["arms"] = {"real": {"topology": "real"}, "none": {"topology": "none"}}
    h["control_arm"] = "none"
    hd, hl = locked(tmp_path, h)
    runs = tmp_path / "runs"
    assert prereg.untrainable_arm(hl, "none") and not prereg.untrainable_arm(hl, "real")
    assert not prereg.untrainable_arm(dict(hl, protocol=dict(PROTOCOL, train_scope="core+top")), "none")
    for s in (1, 2, 3):
        # evaluated as-is (steps_done 0), with the eval recorded at every eval step as run_lab does
        fake_run(runs, hl, "none", s, 1.0, steps=0, untrainable=True)
        fake_run(runs, hl, "real", s, 0.95)
    assert prereg.evaluate_rule(hl, runs, hd)["verdict"] == "SUPPORTED"

    def problems_after(arm, seed, **fields):
        fake_run(runs, hl, arm, seed, 1.0 if arm == "none" else 0.95, **fields)
        res = prereg.evaluate_rule(hl, runs, hd)
        fake_run(runs, hl, arm, seed, 1.0 if arm == "none" else 0.95,
                 **({"steps": 0, "untrainable": True} if arm == "none" else {}))  # restore
        assert (res["verdict"] == "INCONCLUSIVE") == bool(res["problems"])
        return res["problems"]

    # an untrainable run also needs its eval at the final step (run_lab writes one at every eval step)
    probs = problems_after("none", 1, steps=0, untrainable=True, evals=[{"step": 5, "val_nats_per_char_pooled": 1.0}])
    assert probs == ["none seed 1: last eval at step 5, protocol 10"]
    # the untrainable flag is checked against the arm: a trainable arm that claims it (a short run) ...
    probs = problems_after("real", 3, steps=5, untrainable=True, evals=[{"step": 5, "val_nats_per_char_pooled": 0.5}])
    assert len(probs) == 2 and all(p.startswith("real seed 3:") for p in probs)
    assert "untrainable=True" in probs[0] and "last eval at step 5" in probs[1]
    # ... and an untrainable arm recorded as trained
    probs = problems_after("none", 2, steps=10)
    assert probs == ["none seed 2: recorded untrainable=None, but arm 'none' is untrainable under this protocol"]
    # still has to be the right run
    assert problems_after("none", 1, steps=0, untrainable=True, config=cl.LabConfig().to_dict())
    (prereg.run_dir(hl, "none", 1, runs) / "metrics.json").write_text("{not json", encoding="utf-8")
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and "unreadable" in res["problems"][0]
    (prereg.run_dir(hl, "none", 1, runs) / "metrics.json").write_text("[1, 2]", encoding="utf-8")
    res = prereg.evaluate_rule(hl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and "unreadable" in res["problems"][0]


def test_verdict_accepts_runs_shared_with_another_hypothesis(tmp_path):
    """H-B reuses H-A's 'real' runs (same protocol, config and seed = same run key, recorded for H-A)."""
    hd = tmp_path / "hyp"
    a, b = hyp("H-A"), hyp("H-B")
    b["arms"] = {"wired": {"topology": "real"}, "none": {"topology": "none"}}
    b.update(control_arm="none", treatment_arm="wired")
    for h in (a, b):
        write_hyp(hd, h)
        prereg.lock(h["id"], hd)
    al, bl = prereg.load_locked("H-A", hd), prereg.load_locked("H-B", hd)
    runs = tmp_path / "runs"
    assert prereg.run_dir(al, "real", 1, runs) == prereg.run_dir(bl, "wired", 1, runs)
    for s in (1, 2, 3):
        fake_run(runs, al, "real", s, 0.95)  # trained once, for H-A
        fake_run(runs, bl, "none", s, 1.0, steps=0, untrainable=True)
    res = prereg.evaluate_rule(bl, runs, hd)
    assert res["verdict"] == "SUPPORTED", res["problems"]
    # recorded under H-A's name but with H-B's lock: not traceable
    fake_run(runs, al, "real", 2, 0.95, lock=bl["_lock"])
    res = prereg.evaluate_rule(bl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and any("lock" in p for p in res["problems"])
    # recorded for a hypothesis that does not exist
    fake_run(runs, al, "real", 2, 0.95, hypothesis="H-Z")
    res = prereg.evaluate_rule(bl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and any("H-Z" in p for p in res["problems"])
    # recorded for H-A under an arm name whose config is not this arm's
    fake_run(runs, al, "real", 2, 0.95, arm="shuffled")
    assert prereg.evaluate_rule(bl, runs, hd)["verdict"] == "INCONCLUSIVE"
    # recorded for H-A without a lock record (not an --unlocked run): not provably from the locked H-A
    fake_run(runs, al, "real", 2, 0.95, lock=None)
    res = prereg.evaluate_rule(bl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and res["problems"] == ["wired seed 2: no lock recorded: not provably "
                                                                    "made from the locked hypothesis"]
    # recorded for H-C: same protocol and arm config, but H-C was written and never locked
    c = hyp("H-C")
    write_hyp(hd, c)
    assert prereg.run_dir(c, "real", 2, runs) == prereg.run_dir(bl, "wired", 2, runs)
    fake_run(runs, c, "real", 2, 0.95)  # c has no _lock: the run records lock None, unlocked False
    res = prereg.evaluate_rule(bl, runs, hd)
    assert res["verdict"] == "INCONCLUSIVE" and res["problems"] == ["wired seed 2: recorded for H-C, which has no "
                                                                    "readable lock"]
    fake_run(runs, al, "real", 2, 0.95)
    assert prereg.evaluate_rule(bl, runs, hd)["verdict"] == "SUPPORTED"


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


def test_temporal_split_v2():
    n_tr, n_he, seed = 2000, 160, 7
    v1 = tt.split_rows(n_tr, n_he, seed)
    # version 1 is the original split, unchanged (default and explicit)
    held = tt.make_rows(n_he, seed, "heldout")
    orig = [r for r in tt.make_rows(n_tr + n_he, seed, "train") if r["user"] not in {h["user"] for h in held}][:n_tr]
    assert v1 == tt.split_rows(n_tr, n_he, seed, version=1) == {"train": orig, "heldout": held}
    held_users = [r["user"] for r in held]
    held_units = {tt.pattern_unit(r) for r in held} - {None}

    def leaks(rows):
        return [r for r in rows if any(r["user"].startswith(u) for u in held_users) or tt.pattern_unit(r) in held_units]

    assert leaks(v1["train"]), "the v1 split has near-duplicates of held-out rows (why v2 exists)"
    v2 = tt.split_rows(n_tr, n_he, seed, version=2)
    assert v2 == tt.split_rows(n_tr, n_he, seed, version=2)  # deterministic
    assert v2["heldout"] == held and len(v2["train"]) == n_tr
    assert not leaks(v2["train"])
    assert [r["task"] for r in v2["train"][:8]] == list(tt.FAMILIES) * 2  # family round-robin kept
    assert all(v == n_tr // 4 for v in collections.Counter(r["task"] for r in v2["train"]).values())
    # per family: the first train-stream rows that are neither held-out prompts nor near-duplicates of them
    stream = tt.make_rows(4 * (n_tr + n_he), seed, "train")
    for fam in tt.FAMILIES:
        want = [r for r in stream if r["task"] == fam and r["user"] not in set(held_users) and not leaks([r])]
        assert [r for r in v2["train"] if r["task"] == fam] == want[:n_tr // 4]
    # the unit is the minimal period, canonical up to rotation
    u = tt.pattern_unit({"task": "pattern", "user": "Continue the sequence: 4 9 1 4 9 1 4", "assistant": "9 1 4"})
    assert u == ("1", "4", "9") == tt.pattern_unit({"task": "pattern", "user": "Continue the sequence: 9 1 4 9 1",
                                                     "assistant": "4 9"})
    assert tt.pattern_unit({"task": "pattern", "user": "Continue the sequence: 1 2 1 2 1", "assistant": "2 1 2"}) == ("1", "2")
    assert tt.pattern_unit(held[0]) is None or held[0]["task"] == "pattern"
    with pytest.raises(ValueError):
        tt.split_rows(10, 4, seed, version=3)


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


# ---------------------------------------------------------------------------
# run_lab on a tiny synthetic LM (no checkpoint, no corpus)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rl():
    try:
        import run_lab
    except Exception as exc:  # pragma: no cover - run_lab imports the trunk sources (supermix-archimedes)
        pytest.skip(f"run_lab not importable here: {exc}")
    return run_lab


class TinyTok:
    """Char-level stand-in with the BPE tokenizer's turn layout: [BOS, USER] user [ASSISTANT] reply [EOS]."""

    def __init__(self, eos: int):
        self.eos = eos

    @staticmethod
    def encode(s):
        return [6 + (ord(c) % 58) for c in s]

    def encode_turn(self, user, assistant=None):
        ids = [1, 3] + self.encode(user) + [4]
        plen = len(ids)
        if assistant is not None:
            ids = ids + self.encode(assistant) + [self.eos]
        return ids, plen


class TinyLM(nn.Module):
    """Embedding -> causal running mean -> lab core graft (token mask, as the Expanse hook passes it) -> head."""

    def __init__(self, core, vocab=64, hidden=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, hidden, generator=g))
            self.head.weight.copy_(torch.randn(vocab, hidden, generator=g) * 0.5)
        self.cns_full = core
        self.last_graft_info = {}

    def forward(self, x, return_mtp=False, past_length=None, **kw):
        h = self.emb(x)
        h = torch.cumsum(h, 1) / torch.arange(1, x.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        info = {}
        if self.cns_full is not None:
            h, info = self.cns_full(h, x != 0, past_length=past_length)
        self.last_graft_info = {"cns_full": info}
        return SimpleNamespace(logits=self.head(h))


def no_feats(users):
    return {}


def tiny_setup(rl, tmp_path, cfg, seed=0):
    from test_connectome_full import write_tiny_npz

    base = cf.build_full_graph(write_tiny_npz(tmp_path / "tiny.npz", n=40, density=0.2))
    g = cl.lab_graph(base, cfg, 0)
    core = cl.fresh_lab_core(16, g, cfg, seed)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():  # off the exact-identity birth state: a live gate and dynamics
        for name, p in core.named_parameters():
            if name not in ("edge_theta", "edge_alpha"):
                p.add_(torch.randn(p.shape, generator=gen) * 0.3)
    tok = TinyTok(rl.text_utils.EOS)
    rng = random.Random(seed)
    words = "alpha beta gamma delta sigma omega kappa theta".split()
    rows = {s: [{"user": " ".join(rng.choice(words) for _ in range(rng.randint(2, 6))) + "?",
                 "assistant": " ".join(rng.choice(words) for _ in range(rng.randint(1, 4))), "source": s}
                for _ in range(7)] for s in ("code", "bio")}
    rows["temporal"] = tt.make_rows(8, 3, "heldout")
    data = {"eval": {}, "temporal_heldout": []}
    for s, rs in rows.items():
        x, y, _, meta, d = rl.te.encode_rows(rs, tok, 320)
        assert d == 0
        data["eval"][s] = {"x": x, "y": y, "rows": meta, "dropped": d}
    return TinyLM(core, seed=seed), tok, data


def test_run_lab_final_rows_temporal_answer_and_write_stats(rl, tmp_path):
    model, tok, data = tiny_setup(rl, tmp_path, cl.LabConfig(edge_param="dale_softplus"))
    out = {}
    rec = rl.evaluate(model, tok, data, no_feats, batch=3, step=7, row_losses=out)
    for src, block in data["eval"].items():
        rows = out["rows"][src]
        key = "temporal" if src == "temporal" else f"val_{src}"
        assert len(rows) == len(block["rows"])
        assert [r[1] for r in rows] == [len(r["assistant"]) + 1 for r in block["rows"]]
        assert sum(r[2] for r in rows) == int((block["y"][:, 1:] != -100).sum())
        assert sum(r[0] for r in rows) / sum(r[1] for r in rows) == pytest.approx(rec[f"{key}_nats_per_char"], rel=1e-6)
        assert sum(r[0] for r in rows) / sum(r[2] for r in rows) == pytest.approx(rec[f"{key}_nats_per_token"], rel=1e-6)
        keys = "\n".join(rl.ec.row_key(r["user"], r["assistant"]) for r in block["rows"])
        assert out["keys_sha256"][src] == hashlib.sha256(keys.encode()).hexdigest()
    pooled = [sum(r[i] for s in ("code", "bio") for r in out["rows"][s]) for i in (0, 1)]
    assert pooled[0] / pooled[1] == pytest.approx(rec["val_nats_per_char_pooled"], rel=1e-6)
    # answer-only temporal scoring: the end-of-reply token is the one target left out
    ans, full, trows = out["temporal_answer"], out["rows"]["temporal"], data["eval"]["temporal"]["rows"]
    assert [a[2] for a in ans] == [f[2] - 1 for f in full]
    assert [a[1] for a in ans] == [len(r["assistant"]) for r in trows]
    assert all(a[0] < f[0] for a, f in zip(ans, full))
    assert rec["temporal_answer_nats_per_char"] == pytest.approx(sum(a[0] for a in ans) / sum(a[1] for a in ans), rel=1e-6)
    by_task = rec["temporal_nats_per_char_by_task"]
    assert set(by_task) == set(rec["temporal_answer_nats_per_char_by_task"]) == set(tt.FAMILIES)
    for fam in tt.FAMILIES:
        idx = [i for i, r in enumerate(trows) if r["task"] == fam]
        assert by_task[fam] == pytest.approx(sum(full[i][0] for i in idx) / sum(full[i][1] for i in idx), rel=1e-6)
    # write diagnostics: a live core writes something that varies over tokens ...
    assert rec["write_rms_ratio"] > 0 and 0 < rec["write_token_var_frac"] <= 1
    # ... and with every edge zeroed (a static gate) the write is one constant vector
    with rl.ablated_core(model.cns_full, "zero_edges"):
        rec0 = rl.evaluate(model, tok, data, no_feats, batch=3, step=7)
    assert rec0["write_token_var_frac"] < 1e-9 and rec0["write_rms_ratio"] > 0
    # the existing keys are all still there
    for k in ("val_nats_per_char_pooled", "val_nats_per_token_pooled", "temporal_nats_per_char", "gate_abs_mean",
              "gate_by_source", "state_autocorr", "mean_rate", "fraction_active", "eval_seconds"):
        assert k in rec
    metrics = {}
    rl.store_final_rows(metrics, out)
    assert metrics["final_rows"] is out["rows"] and metrics["final_rows_temporal_answer"] is ans
    assert len(json.dumps(metrics)) < 60 * sum(len(v) for v in out["rows"].values()) + 2000  # compact


@pytest.mark.parametrize("cfg", [cl.LabConfig(edge_param="dale_softplus"),
                                 cl.LabConfig(edge_param="dale_softplus", gate="dynamic"),
                                 cl.LabConfig(edge_param="dale_softplus", gate="dynamic", state="persist")],
                         ids=["static-reset", "dynamic-reset", "dynamic-persist"])
def test_run_lab_write_stats_pool_every_eval_token(rl, tmp_path, cfg):
    """write_rms_ratio / write_token_var_frac are over all eval tokens at once (not a mean of per-batch values)."""
    model, tok, data = tiny_setup(rl, tmp_path, cfg)
    core = model.cns_full
    seen = []

    def capture(mod, args, out):  # the core is called as core(hidden, token_mask, past_length=...)
        hidden, mask = args[0], args[1]
        seen.append(((out[0] - hidden)[mask].double(), hidden[mask].double()))

    def brute(records):
        w, hs = torch.cat([a for a, _ in records]), torch.cat([b for _, b in records])
        power = w.pow(2).mean()
        return (float(torch.sqrt(power / hs.pow(2).mean())),
                float((w - w.mean(0, keepdim=True)).pow(2).mean() / power))

    hook = core.register_forward_hook(capture)
    try:
        rec = rl.evaluate(model, tok, data, no_feats, batch=3, step=1)
    finally:
        hook.remove()
    assert len(seen) == sum(-(-b["x"].shape[0] // 3) for b in data["eval"].values())  # every eval forward
    ratio, var = brute(seen)
    assert rec["write_rms_ratio"] == pytest.approx(ratio, rel=1e-4)
    assert rec["write_token_var_frac"] == pytest.approx(var, rel=1e-3)
    # the per-forward values averaged (the old aggregation) are not the eval-level value
    per_fwd = [brute([s]) for s in seen]
    wts = [s[0].shape[0] for s in seen]
    assert abs(sum(wt * v for wt, (_, v) in zip(wts, per_fwd)) / sum(wts) - var) > 1e-3
    # independent of the eval batch size (right-padded rows: every token sees the same inputs)
    for b in (1, 7):
        rb = rl.evaluate(model, tok, data, no_feats, batch=b, step=1)
        assert rb["write_rms_ratio"] == pytest.approx(rec["write_rms_ratio"], rel=1e-5)
        assert rb["write_token_var_frac"] == pytest.approx(rec["write_token_var_frac"], rel=1e-5)
    # edges zeroed: a constant write only with a static gate and state=reset; a dynamic gate still
    # scales the fixed zero-edge output per token
    with rl.ablated_core(core, "zero_edges"):
        rec0 = rl.evaluate(model, tok, data, no_feats, batch=3, step=1)
    if cfg.gate == "static" and cfg.state == "reset":
        assert rec0["write_token_var_frac"] < 1e-9
    else:
        assert rec0["write_token_var_frac"] > 1e-3


def test_run_lab_records_unlocked_for_a_dict_that_was_never_locked(rl):
    h = hyp()
    assert rl.run_provenance(h, False) == {"unlocked": True, "lock": None}
    lock_rec = {"id": h["id"], "sha256": "ab" * 32}
    assert rl.run_provenance(dict(h, _lock=lock_rec), False) == {"unlocked": False, "lock": lock_rec}
    assert rl.run_provenance(dict(h, _lock=lock_rec), True) == {"unlocked": True, "lock": lock_rec}


@pytest.mark.parametrize("cfg", [cl.LabConfig(), cl.LabConfig(edge_param="dale_softplus", gate="dynamic", state="persist")],
                         ids=["fixed-static-reset", "dale-dynamic-persist"])
def test_run_lab_final_ablations_restore_the_core(rl, tmp_path, cfg):
    model, tok, data = tiny_setup(rl, tmp_path, cfg, seed=2)
    core = model.cns_full
    before = copy.deepcopy(core.state_dict())
    x = data["eval"]["code"]["x"][:3]
    with torch.no_grad():
        ref = model(x).logits
    rec = rl.evaluate(model, tok, data, no_feats, batch=3, step=10)
    trained = rec["val_nats_per_char_pooled"]
    assert rl.pooled_nats_per_char(model, data, no_feats, 3) == trained  # the same computation, bit for bit
    ab = rl.final_ablations(model, data, no_feats, 3, trained)
    after = core.state_dict()
    assert list(after) == list(before)
    for k, v in before.items():
        assert after[k].dtype == v.dtype and torch.equal(after[k], v), k
    with torch.no_grad():
        assert torch.equal(model(x).logits, ref)  # no stale operator cache after the restore
    z, g0 = ab["val_nats_per_char_pooled_zero_edges"], ab["val_nats_per_char_pooled_gate0"]
    assert ab["graph_dependent_gain"] == z - trained and ab["core_gain"] == g0 - trained
    assert len({z, g0, trained}) == 3
    model.cns_full = None  # a closed gate is exactly "no core"
    assert rl.pooled_nats_per_char(model, data, no_feats, 3) == g0
    model.cns_full = core
    with pytest.raises(RuntimeError):  # restored even when the ablated evaluation fails
        with rl.ablated_core(core, "gate0"):
            assert not bool(core.gate.detach().any())
            raise RuntimeError("boom")
    assert all(torch.equal(core.state_dict()[k], v) for k, v in before.items())
    with pytest.raises(ValueError):
        with rl.ablated_core(core, "nope"):
            pass
    assert rl.final_ablations(TinyLM(None), data, no_feats, 3, trained) == {}


def test_run_lab_eps_edge_only_for_edges(rl, tmp_path):
    model, _, _ = tiny_setup(rl, tmp_path, cl.LabConfig(edge_param="dale_softplus"))
    core = model.cns_full
    old = rl.trainable_groups(model, core, {"lr_core": 1e-3, "lr_edge": 3e-4})
    assert [g["name"] for g in old] == ["core", "edges"] and all("eps" not in g for g in old)
    new = rl.trainable_groups(model, core, {"lr_core": 1e-3, "lr_edge": 3e-4, "eps_edge": 1e-12})
    assert "eps" not in new[0] and new[1]["eps"] == 1e-12
    for groups, eps_edges in ((old, 1e-8), (new, 1e-12)):
        opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups], betas=(0.9, 0.95))
        assert [pg["eps"] for pg in opt.param_groups] == [1e-8, eps_edges]
        theta0 = core.edge_theta.detach().clone()
        for p in core.parameters():
            p.grad = torch.zeros_like(p)
        core.edge_theta.grad.fill_(1e-9)  # the per-edge gradient size the first sweep saw
        opt.step()
        moved = float((core.edge_theta.detach() - theta0).abs().max())
        with torch.no_grad():
            core.edge_theta.copy_(theta0)
        # first Adam step = lr * g / (|g| + eps): damped ~11x by eps 1e-8, a full lr step with 1e-12
        assert moved == pytest.approx(3e-4 * 1e-9 / (1e-9 + eps_edges), rel=1e-2)


def test_run_lab_eval_sampling_and_temporal_split(rl):
    dev = [{"user": f"question {i}", "assistant": f"answer {i}"} for i in range(50)]
    assert rl.select_eval_rows(dev, 10) == dev[:10] == rl.select_eval_rows(dev, 10, "head", "any salt")
    h1 = rl.select_eval_rows(dev, 10, "hash")
    shuffled = dev[:]
    random.Random(3).shuffle(shuffled)
    assert rl.select_eval_rows(shuffled, 10, "hash") == h1  # independent of file order
    assert len(h1) == 10 and all(r in dev for r in h1) and h1 != dev[:10]
    assert h1 == sorted(dev, key=lambda r: rl.eval_row_hash(r))[:10]
    assert rl.select_eval_rows(dev, 10, "hash", "another salt") != h1
    with pytest.raises(ValueError):
        rl.select_eval_rows(dev, 10, "random")
    P = {"temporal": {"train_rows": 200, "eval_rows": 40, "seed": 7}}
    assert rl.temporal_split(P) == tt.split_rows(200, 40, 7)
    P["temporal"]["split_version"] = 2
    assert rl.temporal_split(P) == tt.split_rows(200, 40, 7, version=2)
    assert rl.temporal_split({}) == {"train": [], "heldout": []}


# ---------------------------------------------------------------------------
# analysis: exploratory arms
# ---------------------------------------------------------------------------
def test_analyze_exploratory_section(tmp_path, monkeypatch):
    import analyze

    hd, runs = tmp_path / "hyp", tmp_path / "runs"
    proto = dict(PROTOCOL, notes="shared")
    pre = hyp("H-P-001")
    pre.update(protocol=proto, arms={"persist": {"state": "persist"}, "reset": {"state": "reset"}},
               control_arm="reset", treatment_arm="persist", primary_metric="temporal_nats_per_char")
    xp = {"id": "X-P-001", "exploratory": True, "claim": "Explore.", "protocol": dict(proto, notes="other notes"),
          "arms": {"reset": {"state": "reset"}, "tanh": {"activation": "tanh"}, "unsigned": {"sign_mode": "unsigned"}},
          "seeds": [1]}
    for h in (pre, xp):
        write_hyp(hd, h)
        prereg.lock(h["id"], hd)
    pl, xl = prereg.load_locked("H-P-001", hd), prereg.load_locked("X-P-001", hd)

    def put(h, arm, seed, pooled, temporal, **extra):
        ev = {"step": 10, "val_nats_per_char_pooled": pooled, "val_code_nats_per_char": pooled + 0.1,
              "temporal_nats_per_char": temporal, **extra}
        d = prereg.run_dir(h, arm, seed, runs)
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps({"status": "done", "hypothesis": h["id"], "arm": arm, "seed": seed,
                                                    "evals": [ev]}), encoding="utf-8")

    # paired persist - reset deltas: pooled (and code) +0.001, +0.003, +0.002 (SD 0.001); temporal all +0.01
    # (SD 0); bio only on seed 1 (one pair: no floor)
    for s, dp in zip((1, 2, 3), (0.001, 0.003, 0.002)):
        put(pl, "reset", s, 0.50, 6.0, **({"val_bio_nats_per_char": 0.7} if s == 1 else {}))
        put(pl, "persist", s, 0.50 + dp, 6.01, **({"val_bio_nats_per_char": 0.7} if s == 1 else {}))
    assert prereg.run_dir(xl, "reset", 1, runs) == prereg.run_dir(pl, "reset", 1, runs)  # shared control
    put(xl, "tanh", 1, 0.5015, 6.5, graph_dependent_gain=0.0012, write_rms_ratio=0.04, write_token_var_frac=0.02)
    put(xl, "unsigned", 1, 0.5100, 6.0, val_bio_nats_per_char=0.75)
    floors, used = analyze.paired_noise_floor(analyze.locked_hypotheses(hd), xl["protocol"], runs,
                                              ("val_nats_per_char_pooled", "temporal_nats_per_char",
                                               "val_bio_nats_per_char"))
    assert used == ["H-P-001"] and floors["val_nats_per_char_pooled"] == pytest.approx(0.001)
    assert floors["temporal_nats_per_char"] == pytest.approx(0.0, abs=1e-12) and "val_bio_nats_per_char" not in floors
    md = "\n".join(analyze.exploratory_section(hd, runs))
    assert "## Exploratory arms: X-P-001" in md and "single seed, no claims" in md.lower()
    assert "Reference arm: `reset`" in md and "H-P-001" in md

    def delta_cells(text, arm):  # columns: pooled replay code bio connectome fresh temporal temporal-answer
        line = [ln for ln in text.splitlines() if ln.startswith(f"| {arm} | 1 | ") and "done" not in ln][0]
        return [c.strip() for c in line.split("|")[3:-1]]

    tanh = delta_cells(md, "tanh")
    assert tanh[0] == "+0.00150" and tanh[6] == "+0.50000*"  # pooled within 2 x 0.001; temporal beyond 0
    unsigned = delta_cells(md, "unsigned")
    assert unsigned[0] == "+0.01000*" and unsigned[2] == "+0.01000*"  # beyond 2 x 0.001
    assert unsigned[1] == "" and unsigned[3] == "+0.05000?"  # replay not recorded; bio has no floor
    assert unsigned[6] == "+0.00000"
    assert "graph-dependent gain" in md and "0.00120" in md  # diagnostics listed when present
    md2 = "\n".join(analyze.exploratory_section(hd, runs, noise_floor=0.01))
    assert "passed in" in md2
    assert delta_cells(md2, "unsigned")[0] == "+0.01000"  # |delta| = 1 x floor: not flagged at k = 2
    assert delta_cells(md2, "unsigned")[3] == "+0.05000*"
    monkeypatch.setattr(prereg, "VERDICTS", tmp_path / "no_verdicts")
    full = analyze.summary([json.loads((prereg.run_dir(xl, "tanh", 1, runs) / "metrics.json").read_text())], hd, runs)
    assert "Final-eval ablations and core-write diagnostics" in full and "Exploratory arms" in full
