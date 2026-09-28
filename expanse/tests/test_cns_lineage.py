"""Input-byte cache validation and ordered cohort receipts, without model jobs."""

import copy
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cns_lab"))
import lineage


def test_manifest_order_and_duplicates_do_not_change_identity(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "nested" / "b.jsonl"
    b.parent.mkdir()
    a.write_bytes(b"checkpoint")
    b.write_bytes(b"corpus")
    first = lineage.input_manifest([b, a, a], tmp_path)
    assert first == lineage.input_manifest([a, b], tmp_path)
    assert [entry["path"] for entry in first["files"]] == ["a.pt", "nested/b.jsonl"]
    assert lineage.validate_input_manifest(first) == []
    # Timestamps are deliberately not an input identity.
    os.utime(a, (1_000_000, 1_000_000))
    assert lineage.input_manifest([a, b], tmp_path) == first


def test_changed_bytes_invalidate_cache_even_with_same_size_and_mtime(tmp_path):
    source = tmp_path / "model.py"
    source.write_bytes(b"alpha")
    stat = source.stat()
    original = lineage.input_manifest([source], tmp_path)
    source.write_bytes(b"bravo")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = lineage.input_manifest([source], tmp_path)
    assert changed["sha256"] != original["sha256"]
    with pytest.raises(RuntimeError, match="stale-cache"):
        lineage.require_reusable_run({"input_manifest": original}, changed)


def test_valid_cache_reuses_and_legacy_or_tampered_cache_refuses(tmp_path):
    source = tmp_path / "base.pt"
    source.write_bytes(b"base")
    manifest = lineage.input_manifest([source], tmp_path)
    lineage.require_reusable_run({"input_manifest": manifest}, copy.deepcopy(manifest))
    with pytest.raises(RuntimeError, match="unverifiable-cache.*legacy"):
        lineage.require_reusable_run({"status": "done"}, manifest)
    bad = copy.deepcopy(manifest)
    bad["files"][0]["sha256"] = "0" * 64
    assert "aggregate sha256 mismatch" in lineage.validate_input_manifest(bad)
    with pytest.raises(RuntimeError, match="unverifiable-cache"):
        lineage.require_reusable_run({"input_manifest": bad}, manifest)
    with pytest.raises(FileNotFoundError):
        lineage.input_manifest([tmp_path / "absent.pt"], tmp_path)


def test_source_inventory_is_bounded_and_honors_omni_override(tmp_path, monkeypatch):
    names = ["expanse/train_expanse.py", "expanse/src/model.py", "expanse/cns_lab/runner.py",
             "supermix-archimedes/archimedes/src/champion/model.py",
             "supermix-archimedes/models/supermix-v93/src/text.py", "custom_omni/encoder.py"]
    excluded = ["expanse/tests/test_model.py", "expanse/checkpoints/generated.py",
                "external/omni_v7/unused.py", "custom_omni/site-packages/dependency.py"]
    for name in names + excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source", encoding="utf-8")
    monkeypatch.setenv("EXPANSE_OMNI_V7_DIR", str(tmp_path / "custom_omni"))
    found = {p.relative_to(tmp_path).as_posix() for p in lineage.project_source_files(tmp_path)}
    assert found == set(names)


def _data():
    row = lambda user: {"user": user, "assistant": "answer", "source": "code"}
    return {"m_tr": [row("a"), row("b")], "m_tt": [],
            "eval": {"code": {"rows": [row("c"), row("d")]}}, "temporal_heldout": []}


def test_data_lineage_binds_membership_order_and_batch_plan():
    data = _data()
    plan = [{"corpus": [1, 0], "temporal": None}]
    receipt = lineage.data_lineage(data, plan)
    assert lineage.validate_data_lineage(receipt) == []
    reversed_train = copy.deepcopy(data)
    reversed_train["m_tr"].reverse()
    assert lineage.data_lineage(reversed_train, plan)["train"] != receipt["train"]
    different_eval = copy.deepcopy(data)
    different_eval["eval"]["code"]["rows"][0]["assistant"] = "changed"
    assert lineage.data_lineage(different_eval, plan)["eval"] != receipt["eval"]
    changed_plan = [{"corpus": [0, 1], "temporal": None}]
    assert lineage.data_lineage(data, changed_plan)["batch_plan"] != receipt["batch_plan"]
    bad = copy.deepcopy(receipt)
    bad["train"]["count"] = True
    assert lineage.validate_data_lineage(bad)


@pytest.fixture
def runner():
    sys.path.insert(0, str(ROOT / "src"))
    return importlib.import_module("run_lab")


def _finished_metrics(runner, hyp, protocol, cfg, directory, manifest, *, unlocked=False):
    data = _data()
    rows = data["eval"]["code"]["rows"]
    return {"schema": runner.METRICS_SCHEMA, "status": "done", "hypothesis": hyp["id"],
            "arm": "real", "seed": 1, "run_key": directory.name, "protocol": protocol,
            "config": cfg.to_dict(), "steps_done": protocol["steps"], "untrainable": False,
            "unlocked": unlocked, "lock": hyp.get("_lock"), "input_manifest": manifest,
            "data_lineage": lineage.data_lineage(data, [{"corpus": [0], "temporal": None}]),
            "evals": [{"step": protocol["steps"], "val_nats_per_char_pooled": 0.4}],
            "final_rows": {"code": [[1.0, 7, 1] for _ in rows]},
            "final_rows_sha256": {"code": lineage.canonical_sha256(rows)}}


@pytest.mark.parametrize("kind", ["valid", "legacy", "stale", "missing"])
def test_runner_checks_finished_cache_before_loading_model(tmp_path, monkeypatch, runner, kind):
    inputs = tmp_path / "inputs.pt"
    inputs.write_bytes(b"v1")
    manifest = lineage.input_manifest([inputs], tmp_path)
    hyp = {"id": "X-CACHE", "protocol": {"steps": 1}, "arms": {"real": {}},
           "_lock": {"id": "X-CACHE", "sha256": "a" * 64}}
    cfg = runner.prereg.arm_config(hyp, "real")
    directory = tmp_path / "runs" / runner.prereg.run_key(hyp["protocol"], cfg, 1)
    directory.mkdir(parents=True)
    metrics = _finished_metrics(runner, hyp, hyp["protocol"], cfg, directory, manifest)
    if kind == "legacy":
        del metrics["input_manifest"]
    path = directory / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    before = path.read_bytes()
    if kind == "stale":
        inputs.write_bytes(b"v2")
    elif kind == "missing":
        inputs.unlink()
    monkeypatch.setattr(runner, "run_input_manifest", lambda _: lineage.input_manifest([inputs], tmp_path))
    monkeypatch.setattr(runner.ec, "load_expanse", lambda *_: pytest.fail("cache check loaded a model"))
    if kind == "valid":
        assert runner.run(hyp, "real", 1, runs_root=tmp_path / "runs") == metrics
    else:
        with pytest.raises(RuntimeError, match="stale-cache" if kind == "stale" else "unverifiable-cache"):
            runner.run(hyp, "real", 1, runs_root=tmp_path / "runs")
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["skeleton", "protocol", "config", "seed", "run_key", "steps",
                                     "no_eval", "bad_eval", "no_data", "no_rows", "lock", "unlocked"])
def test_runner_rejects_incomplete_or_wrong_identity_cache(tmp_path, monkeypatch, runner, mutation):
    inputs = tmp_path / "inputs.pt"
    inputs.write_bytes(b"v1")
    manifest = lineage.input_manifest([inputs], tmp_path)
    hyp = {"id": "X-CACHE", "protocol": {"steps": 1}, "arms": {"real": {}},
           "_lock": {"id": "X-CACHE", "sha256": "a" * 64}}
    cfg = runner.prereg.arm_config(hyp, "real")
    directory = tmp_path / runner.prereg.run_key(hyp["protocol"], cfg, 1)
    directory.mkdir()
    metrics = _finished_metrics(runner, hyp, copy.deepcopy(hyp["protocol"]), cfg, directory, manifest)
    if mutation == "skeleton":
        metrics = {"status": "done", "input_manifest": manifest}
    elif mutation == "protocol":
        metrics["protocol"]["batch"] = 12
    elif mutation == "config":
        metrics["config"]["topology"] = "none"
    elif mutation == "seed":
        metrics["seed"] = True
    elif mutation == "run_key":
        metrics["run_key"] = "other"
    elif mutation == "steps":
        metrics["steps_done"] = 1.0
    elif mutation == "no_eval":
        metrics["evals"] = []
    elif mutation == "bad_eval":
        metrics["evals"][-1]["step"] = 0
    elif mutation == "no_data":
        del metrics["data_lineage"]
    elif mutation == "no_rows":
        del metrics["final_rows"]
    elif mutation == "lock":
        metrics["lock"] = {"sha256": "b" * 64}
    elif mutation == "unlocked":
        metrics["unlocked"] = True
    path = directory / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(runner, "run_input_manifest", lambda _: manifest)
    monkeypatch.setattr(runner.ec, "load_expanse", lambda *_: pytest.fail("invalid cache loaded model"))
    with pytest.raises(RuntimeError, match="unverifiable-cache"):
        runner.run(hyp, "real", 1, runs_root=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("smoke, unlocked", [(True, False), (True, True), (False, True)])
def test_smoke_and_explicit_unlocked_cache_reuse(tmp_path, monkeypatch, runner, smoke, unlocked):
    inputs = tmp_path / "inputs.pt"
    inputs.write_bytes(b"v1")
    manifest = lineage.input_manifest([inputs], tmp_path)
    hyp = {"id": "X-CACHE", "protocol": {"steps": 250}, "arms": {"real": {}}}
    if not unlocked:
        hyp["_lock"] = {"id": "X-CACHE", "sha256": "a" * 64}
    protocol = dict(hyp["protocol"])
    if smoke:
        protocol.update({"steps": 3, "eval_steps": [3], "eval_per_source": 4,
                         "temporal": {"eval_rows": 8, "train_rows": 16, "exact_match_items": 4}})
    cfg = runner.prereg.arm_config(hyp, "real")
    directory = tmp_path / (runner.prereg.run_key(protocol, cfg, 1) + ("_smoke" if smoke else ""))
    directory.mkdir()
    metrics = _finished_metrics(runner, hyp, protocol, cfg, directory, manifest, unlocked=unlocked)
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(runner, "run_input_manifest", lambda _: manifest)
    monkeypatch.setattr(runner.ec, "load_expanse", lambda *_: pytest.fail("valid cache loaded model"))
    assert runner.run(hyp, "real", 1, runs_root=tmp_path, smoke=smoke, unlocked=unlocked) == metrics


def test_runner_inventory_uses_selected_v3_inputs_and_optional_features(tmp_path, monkeypatch, runner):
    expanse = tmp_path / "expanse"
    paths = {"data": expanse / "data", "corpus": tmp_path / "replay",
             "code_rows": expanse / "data/code.jsonl", "bio_rows": expanse / "data/bio.clean.jsonl"}
    selected = {"connectome": expanse / "data/v3/connectome.jsonl",
                "fresh": [expanse / "data/v3/fresh_numbers.jsonl"]}
    graph = expanse / "data/graph.npz"
    required = [expanse / "base.pt", graph, paths["code_rows"], paths["bio_rows"],
                selected["connectome"], *selected["fresh"],
                *(paths["corpus"] / f"{name}.jsonl" for name in ("omni", "code", "math"))]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")
    monkeypatch.setattr(runner, "ROOT", expanse)
    monkeypatch.setattr(runner, "PATHS", paths)
    monkeypatch.setattr(runner.cl, "npz_path", lambda: graph)
    calls = []
    monkeypatch.setattr(runner.te, "corpus_files", lambda **kw: calls.append(kw) or selected)
    first = runner.run_input_manifest({"base_checkpoint": "base.pt"})
    assert calls == [{"v3": True}]
    assert {entry["path"] for entry in first["files"]} == {p.relative_to(tmp_path).as_posix() for p in required}
    cache = paths["data"] / "omni7_states.pt"
    cache.write_bytes(b"cached features")
    second = runner.run_input_manifest({"base_checkpoint": "base.pt"})
    assert second["sha256"] != first["sha256"]
    assert "expanse/data/omni7_states.pt" in {entry["path"] for entry in second["files"]}


def test_run_all_routes_done_cache_through_runner_and_skips_failed_verdict(tmp_path, monkeypatch):
    sweep = importlib.import_module("run_all")
    key = "completed-key"
    directory = tmp_path / key
    directory.mkdir()
    (directory / "metrics.json").write_text('{"status":"done"}', encoding="utf-8")
    hyp = {"seeds": [1], "control_arm": "c", "treatment_arm": "t"}
    monkeypatch.setattr(sweep, "LOG", tmp_path / "sweep.log")
    monkeypatch.setattr(sweep.prereg, "RUNS", tmp_path)
    monkeypatch.setattr(sweep, "plan", lambda *_: [("H", "c", 1, key)])
    monkeypatch.setattr(sweep.prereg, "load_locked", lambda _: hyp)
    monkeypatch.setattr(sweep.prereg, "run_dir", lambda *_: directory)
    monkeypatch.setattr(sweep.prereg, "verdict", lambda *_: pytest.fail("failed cached run reached verdict"))
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if "run_lab.py" in command[1] else 0)

    monkeypatch.setattr(sweep.subprocess, "run", invoke)
    assert sweep.main(["--hypotheses", "H"]) == 1
    assert any(Path(command[1]).name == "run_lab.py" for command in calls)
