"""Study evidence cannot silently survive changed inputs or mismatched arms."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXPANSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPANSE))
sys.path.insert(0, str(EXPANSE / "src"))
import run_mixture_study as study
from reasoning_benchmark import make_suite


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    project = tmp_path / "project"
    root = project / "expanse"
    root.mkdir(parents=True)
    monkeypatch.setattr(study, "PROJECT", project)
    monkeypatch.setattr(study, "ROOT", root)
    parent, suite = root / "parent.pt", root / "suite.json"
    parent.write_bytes(b"frozen parent checkpoint")
    suite.write_text("placeholder", encoding="utf-8")
    (root / "model.py").write_text("# frozen source\n", encoding="utf-8")
    for path in study.bound_inputs(parent, suite):
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"user":"unrelated corpus question","assistant":"reply"}\n', encoding="utf-8")
    corpus = project / "supermix-archimedes" / "corpus" / "omni.jsonl"
    suite.write_text(json.dumps(make_suite([corpus], groups_per_family=1)), encoding="utf-8")
    initial = root / "data" / "omni7_states.pt"
    initial.write_bytes(b"frozen features")
    directory = tmp_path / "study"
    study.prepare(SimpleNamespace(directory=directory, parent=parent, suite=suite,
                                  steps=8, threads=2))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return directory, manifest


def training_receipt(arm, manifest):
    sampling = {"schema": "expanse-source-sampler-v1", "mode": arm, "seed": 2026,
                "row_identity": "c" * 64,
                "source_row_probabilities": {"replay": 0.5, "fresh": 0.5},
                "pool_rows": {"replay": 4, "fresh": 4}}
    arguments = {"steps": 8, "batch": 4, "seq": 128, "lr_trunk": 1e-5,
                 "lr_graft": 1e-4, "kd_arch": 0.0, "kd_T": 2.0, "kd_topk": 32,
                 "omni7_kd": 0.2, "fly_aux": 1.0, "wake": "preserve",
                 "freeze_trunk_steps": 0, "boost_all_embeddings": False,
                 "v2_balance": 0.01, "seed": 2026, "threads": 2,
                 "eval_every": 8, "smoke": False}
    parent_hash = study.file_sha256(Path(manifest["parent"]))
    return {**arguments, "target_steps": 8, "ended_by": "done",
            "parent_sha256": parent_hash, "data_fingerprint": "b" * 64,
            "dev_keys": {"replay": ["q1"], "fresh": ["q2"]},
            "packed": {"train": 8, "dev": 2, "dropped": 0},
            "corpus_args": {"fly_rows": 1500, "dev_frac": 0.05, "dev_cap": 24,
                            "max_rows_per_source": 12000, "seed": 2026, "v3_data": True,
                            "v3_dir": str(study.ROOT / "data" / "v3")},
            "sampling": {**sampling, "weight_unit": "row_probability",
                         "exposure": {"complete": True, "since_step": 0, "updates": 8,
                                      "rows": {"replay": 16, "fresh": 16},
                                      "supervised_tokens": {"replay": 32, "fresh": 48}}},
            "resume_contract": {"schema": "expanse-training-contract-v1",
                                "parent_sha256": parent_hash, "data_fingerprint": "b" * 64,
                                "sampling": sampling, "arguments": arguments}}


def write_completed(directory, arm, manifest, training=None):
    checkpoint = directory / f"{arm}.pt"
    receipt = directory / f"{arm}.receipt.json"
    cache = directory / f"omni7.{arm}.pt"
    checkpoint.write_bytes(f"trained {arm}".encode())
    receipt.write_text(json.dumps({"stage": "trained", "training":
                                  training or training_receipt(arm, manifest)}), encoding="utf-8")
    cache.write_bytes(b"arm features")
    run = {"status": "complete", "manifest_sha256": manifest["sha256"],
           "command": manifest["commands"][arm],
           "checkpoint_sha256": study.file_sha256(checkpoint),
           "receipt_sha256": study.file_sha256(receipt),
           "feature_cache_sha256": study.file_sha256(cache), "elapsed_seconds": 1.0}
    (directory / f"{arm}.run.json").write_text(json.dumps(run), encoding="utf-8")


def test_frozen_manifest_accepts_identical_inputs(prepared):
    _, manifest = prepared
    study.verify_manifest(manifest)


@pytest.mark.parametrize("changed", ["parent", "suite", "source", "corpus", "cache"])
def test_manifest_refuses_changed_inputs_without_rewriting(prepared, changed):
    directory, manifest = prepared
    paths = {"parent": Path(manifest["parent"]), "suite": Path(manifest["suite"]),
             "source": study.ROOT / "model.py",
             "corpus": study.PROJECT / "supermix-archimedes" / "corpus" / "omni.jsonl",
             "cache": Path(manifest["initial_cache"])}
    before = (directory / "manifest.json").read_bytes()
    paths[changed].write_bytes(b"changed bytes")
    with pytest.raises(ValueError):
        study.verify_manifest(manifest)
    assert (directory / "manifest.json").read_bytes() == before


def test_manifest_refuses_added_source_file(prepared):
    _, manifest = prepared
    (study.ROOT / "new_helper.py").write_text("# new runtime source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory changed"):
        study.verify_manifest(manifest)


def test_manifest_refuses_edited_design(prepared):
    _, manifest = prepared
    manifest["design"]["steps_per_arm"] += 1
    with pytest.raises(ValueError, match="Invalid or edited"):
        study.verify_manifest(manifest)


def test_new_artifacts_and_prepare_never_overwrite(prepared):
    directory, _ = prepared
    path = directory / "manifest.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        study.write_new(path, {"replacement": True})
    with pytest.raises(ValueError, match="empty directory"):
        study.prepare(SimpleNamespace(directory=directory))
    assert path.read_bytes() == before


def test_train_refuses_unreceipted_checkpoint(prepared, monkeypatch):
    directory, _ = prepared
    checkpoint = directory / "natural.pt"
    checkpoint.write_bytes(b"unreceipted artifact")
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **k: pytest.fail("trainer must not launch"))
    with pytest.raises(ValueError, match="Unreceipted"):
        study.train(directory)
    assert checkpoint.read_bytes() == b"unreceipted artifact"


@pytest.mark.parametrize("changed", ["checkpoint", "receipt", "feature_cache"])
def test_completed_arm_refuses_changed_artifact(prepared, changed):
    directory, manifest = prepared
    write_completed(directory, "natural", manifest)
    paths = {"checkpoint": directory / "natural.pt", "receipt": directory / "natural.receipt.json",
             "feature_cache": directory / "omni7.natural.pt"}
    paths[changed].write_bytes(b"changed artifact")
    with pytest.raises(ValueError):
        study.verify_completed(directory, "natural", manifest)


@pytest.mark.parametrize("changed", ["status", "command", "manifest_sha256"])
def test_completed_arm_refuses_mismatched_run_receipt(prepared, changed):
    directory, manifest = prepared
    write_completed(directory, "natural", manifest)
    path = directory / "natural.run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run[changed] = "wrong"
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="Unverifiable"):
        study.verify_completed(directory, "natural", manifest)


@pytest.mark.parametrize("changed", ["steps", "target_steps", "parent_sha256", "sampling_mode",
                                     "sampling_seed", "exposure", "lr", "corpus"])
def test_completed_arm_refuses_wrong_training_contract(prepared, changed):
    directory, manifest = prepared
    training = training_receipt("natural", manifest)
    if changed in ("steps", "target_steps"):
        training[changed] -= 1
    elif changed == "parent_sha256":
        training[changed] = "e" * 64
        training["resume_contract"][changed] = "e" * 64
    elif changed == "sampling_mode":
        training["sampling"]["mode"] = "balanced"
        training["resume_contract"]["sampling"]["mode"] = "balanced"
    elif changed == "sampling_seed":
        training["sampling"]["seed"] = 7
        training["resume_contract"]["sampling"]["seed"] = 7
    elif changed == "exposure":
        training["sampling"]["exposure"]["complete"] = False
    elif changed == "lr":
        training["lr_trunk"] = training["resume_contract"]["arguments"]["lr_trunk"] = 0.1
    else:
        training["corpus_args"]["fly_rows"] = 4000
    write_completed(directory, "natural", manifest, training)
    with pytest.raises(ValueError):
        study.verify_completed(directory, "natural", manifest)


def test_both_completed_arms_are_reused_without_launch(prepared, monkeypatch):
    directory, manifest = prepared
    for arm in study.ARMS:
        write_completed(directory, arm, manifest)
    before = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **k: pytest.fail("trainer must not launch"))
    study.train(directory)
    assert before == {path: path.read_bytes() for path in before}


def test_paired_arms_must_have_identical_packed_data(prepared, monkeypatch):
    directory, manifest = prepared
    write_completed(directory, "natural", manifest)
    training = training_receipt("balanced", manifest)
    training["data_fingerprint"] = "d" * 64
    training["resume_contract"]["data_fingerprint"] = "d" * 64
    write_completed(directory, "balanced", manifest, training)
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **k: pytest.fail("trainer must not launch"))
    with pytest.raises(ValueError, match="Matched-arm invariant.*data_fingerprint"):
        study.train(directory)


def test_each_launch_reconstructs_cache_from_bound_initial_bytes(prepared, monkeypatch):
    directory, manifest = prepared
    initial = Path(manifest["initial_cache"]).read_bytes()
    for arm in study.ARMS:
        (directory / f"omni7.{arm}.pt").write_bytes(b"untrusted stale features")
    launched = []

    def fake_trainer(command, **kwargs):
        arm = command[command.index("--sampling") + 1]
        cache = Path(command[command.index("--omni7_cache") + 1])
        assert cache.read_bytes() == initial
        launched.append(arm)
        (directory / f"{arm}.pt").write_bytes(f"trained {arm}".encode())
        (directory / f"{arm}.receipt.json").write_text(
            json.dumps({"stage": "trained", "training": training_receipt(arm, manifest)}), encoding="utf-8")
        cache.write_bytes(initial + b"deterministic extra rows")
        return SimpleNamespace(stdout=["fake trainer complete\n"], wait=lambda: 0)

    monkeypatch.setattr(study.subprocess, "Popen", fake_trainer)
    study.train(directory)
    assert launched == list(study.ARMS)
    for arm in study.ARMS:
        study.verify_completed(directory, arm, manifest)


def test_failed_trainer_never_gets_completion_receipt(prepared, monkeypatch):
    directory, _ = prepared
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **k:
                        SimpleNamespace(stdout=["failed before checkpoint\n"], wait=lambda: 3))
    with pytest.raises(RuntimeError, match="trainer exited 3"):
        study.train(directory)
    assert not (directory / "natural.run.json").exists()
    assert not (directory / "balanced.log").exists()
