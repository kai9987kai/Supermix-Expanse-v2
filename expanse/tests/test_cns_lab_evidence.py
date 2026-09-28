"""Adversarial verdict regressions: exact protocol and paired evaluation evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_cns_lab import hyp, locked, prereg, supported_runs, write_hyp


@pytest.fixture
def evidence(tmp_path):
    hd, h = locked(tmp_path, hyp())
    runs = tmp_path / "runs"
    supported_runs(runs, h)
    return hd, h, runs


def edit_run(evidence, arm="real", seed=1, **fields):
    _, h, runs = evidence
    path = prereg.run_dir(h, arm, seed, runs) / "metrics.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    m.update(fields)
    path.write_text(json.dumps(m), encoding="utf-8")
    return path


def evaluate(evidence):
    hd, h, runs = evidence
    return prereg.evaluate_rule(h, runs, hd)


@pytest.mark.parametrize("protocol", [None, {}, "claimed protocol", {"steps": 10, "batch": True},
    {"steps": 10, "batch": 3, "eval_steps": [5, 10]},
    {"steps": 10, "batch": 2, "eval_steps": [5, 10], "eval_salt": "different cohort"}])
def test_protocol_cannot_be_forged_while_retaining_run_key(evidence, protocol):
    edit_run(evidence, protocol=protocol)
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("protocol" in p for p in result["problems"])


def test_protocol_missing_is_not_a_proven_run(evidence):
    path = edit_run(evidence)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    del metrics["protocol"]
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert evaluate(evidence)["verdict"] == "INCONCLUSIVE"


def test_protocol_notes_and_folded_base_config_do_not_change_identity(evidence):
    _, h, _ = evidence
    protocol = {k: v for k, v in h["protocol"].items() if k != "base_config"}
    protocol["notes"] = "An annotation, not a training parameter."
    edit_run(evidence, protocol=protocol)
    assert evaluate(evidence)["verdict"] == "SUPPORTED"


def test_matched_row_hashes_and_legacy_missing_receipts(evidence):
    assert evaluate(evidence)["verdict"] == "SUPPORTED"
    hashes = {"replay": "a" * 64, "temporal": "b" * 64}
    for arm in ("real", "shuffled"):
        edit_run(evidence, arm, final_rows_sha256=hashes)
    assert evaluate(evidence)["verdict"] == "SUPPORTED"


@pytest.mark.parametrize("left,right", [
    ({"replay": "a" * 64}, {"replay": "b" * 64}),
    ({"replay": "a" * 64}, {"code": "a" * 64}),
    ({"replay": "a" * 64}, {"replay": "a" * 64, "temporal": "b" * 64}),
])
def test_equal_metrics_on_different_eval_cohorts_cannot_support(evidence, left, right):
    edit_run(evidence, "real", final_rows_sha256=left)
    edit_run(evidence, "shuffled", final_rows_sha256=right)
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("final_rows_sha256" in p for p in result["problems"])


@pytest.mark.parametrize("arm", ["real", "shuffled"])
def test_one_missing_hash_receipt_is_not_legacy_pair(evidence, arm):
    edit_run(evidence, arm, final_rows_sha256={"replay": "a" * 64})
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("final_rows_sha256" in p for p in result["problems"])


@pytest.mark.parametrize("hashes", [None, [], {}, "a" * 64, {"replay": None},
    {"replay": "a" * 63}, {"replay": "z" * 64}, {"": "a" * 64}, {"replay": True}])
def test_identical_invalid_hash_receipts_do_not_prove_pairing(evidence, hashes):
    for arm in ("real", "shuffled"):
        edit_run(evidence, arm, final_rows_sha256=hashes)
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("final_rows_sha256" in p for p in result["problems"])


def test_shared_run_owner_protocol_must_match(evidence):
    hd, h, _ = evidence
    owner = copy.deepcopy(hyp("H-OWNER"))
    owner["protocol"]["batch"] = 99
    write_hyp(hd, owner)
    lock = prereg.lock(owner["id"], hd)
    edit_run(evidence, hypothesis=owner["id"], lock=lock)
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("protocol" in p for p in result["problems"])


def test_shared_hypothesis_content_must_still_match_its_lock(evidence):
    hd, _, _ = evidence
    owner = copy.deepcopy(hyp("H-OWNER"))
    write_hyp(hd, owner)
    lock = prereg.lock(owner["id"], hd)
    edit_run(evidence, hypothesis=owner["id"], lock=lock)
    assert evaluate(evidence)["verdict"] == "SUPPORTED"
    owner["claim"] = "Edited after the experiment was locked."
    write_hyp(hd, owner)
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("lock" in p for p in result["problems"])


def test_issued_verdict_is_never_rewritten_after_stricter_checks(evidence, tmp_path):
    hd, h, runs = evidence
    verdict_dir = tmp_path / "verdicts"
    issued = prereg.verdict(h["id"], hd, runs, verdict_dir)
    path = verdict_dir / f"{h['id']}.verdict.json"
    original = path.read_bytes()
    edit_run(evidence, final_rows_sha256={"replay": "a" * 64})
    assert evaluate(evidence)["verdict"] == "INCONCLUSIVE"
    assert prereg.verdict(h["id"], hd, runs, verdict_dir)["verdict"] == issued["verdict"]
    assert path.read_bytes() == original


def receipt(field, digest="a" * 64):
    if field == "input_manifest":
        body = {"schema": "supermix-expanse-cns-lab-inputs-v1",
                "files": [{"path": "expanse/checkpoints/base.pt", "sha256": digest}]}
    else:
        row = {"count": 2, "sha256": digest}
        body = {"schema": "supermix-expanse-cns-lab-data-v1", "train": row,
                "temporal_train": row, "eval": {"replay": row}, "temporal_heldout": row,
                "batch_plan": {"count": 10, "sha256": digest}}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return {**body, "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}


@pytest.mark.parametrize("field", ["input_manifest", "data_lineage"])
def test_content_bound_receipts_match_and_report_legacy_limits(evidence, field):
    legacy = evaluate(evidence)
    assert legacy["per_seed"][0]["paired_evidence"][field] == "legacy_unverified"
    for arm in ("real", "shuffled"):
        edit_run(evidence, arm, **{field: receipt(field)})
    result = evaluate(evidence)
    assert result["verdict"] == "SUPPORTED"
    assert result["per_seed"][0]["paired_evidence"][field] == "matched"


@pytest.mark.parametrize("field", ["input_manifest", "data_lineage"])
@pytest.mark.parametrize("arm", ["real", "shuffled"])
def test_content_bound_receipt_missing_on_one_arm_blocks_support(evidence, field, arm):
    edit_run(evidence, arm, **{field: receipt(field)})
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(field in p and "only one arm" in p for p in result["problems"])


@pytest.mark.parametrize("field", ["input_manifest", "data_lineage"])
def test_different_valid_content_receipts_block_support(evidence, field):
    edit_run(evidence, "real", **{field: receipt(field)})
    edit_run(evidence, "shuffled", **{field: receipt(field, "b" * 64)})
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(field in p and "differs" in p for p in result["problems"])
    assert "delta" not in result["per_seed"][0]


@pytest.mark.parametrize("field", ["input_manifest", "data_lineage"])
@pytest.mark.parametrize("bad", [None, {}, "digest", {"schema": "forged", "sha256": "a" * 64}])
def test_equal_invalid_content_receipts_do_not_prove_pairing(evidence, field, bad):
    for arm in ("real", "shuffled"):
        edit_run(evidence, arm, **{field: bad})
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(field in p for p in result["problems"])


@pytest.mark.parametrize("field", ["input_manifest", "data_lineage"])
def test_receipt_with_forged_aggregate_hash_fails_closed(evidence, field):
    manifest = receipt(field)
    manifest["sha256"] = "c" * 64
    for arm in ("real", "shuffled"):
        edit_run(evidence, arm, **{field: manifest})
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(field in p and "aggregate" in p for p in result["problems"])


def test_row_hash_sources_must_describe_recorded_losses(evidence):
    for arm in ("real", "shuffled"):
        edit_run(evidence, arm, final_rows={"replay": [[0.4, 1, 1]], "code": [[0.2, 1, 1]]},
                 final_rows_sha256={"replay": "a" * 64})
    result = evaluate(evidence)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("sources differ" in p for p in result["problems"])
