"""Pilot reporting must refuse an unpaired or incomplete experiment."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cns_lab"))
sys.path.insert(0, str(ROOT / "tests"))

import lineage
import prereg
import report_pilot
from test_cns_lab import fake_run, hyp, write_hyp


@pytest.fixture
def pilot(tmp_path):
    hd, runs = tmp_path / "hyp", tmp_path / "runs"
    h = copy.deepcopy(hyp())
    h["exploratory"] = True
    write_hyp(hd, h)
    prereg.lock(h["id"], hd)
    h = prereg.load_locked(h["id"], hd)
    source = tmp_path / "input.txt"
    source.write_text("frozen input")
    manifest = lineage.input_manifest([source], tmp_path)
    data = {"m_tr": [{"user": "q", "assistant": "a"}], "m_tt": [],
            "eval": {"replay": {"rows": [{"user": "v", "assistant": "a"}]}}, "temporal_heldout": []}
    dl = lineage.data_lineage(data, [{"corpus": [0], "temporal": None}])
    for seed in h["seeds"]:
        for arm in h["arms"]:
            fake_run(runs, h, arm, seed, 0.4 if arm == "real" else 0.5,
                     input_manifest=manifest, data_lineage=dl, final_rows_sha256={"replay": "a" * 64})
    return hd, h, runs


def test_pilot_reports_descriptive_results_without_promotion(pilot):
    hd, h, runs = pilot
    report = report_pilot.summarize(h, runs, hd)
    assert report["promotion_allowed"] is False
    assert report["status"] == "exploratory_only"
    assert report["means"]["real"]["mean_nats_per_char"] == pytest.approx(0.4)
    assert len(report["inputs"]) == 6
    assert "No model was promoted" in report_pilot.markdown(report)


@pytest.mark.parametrize("change", ["cohort", "incomplete", "nonfinite", "lineage", "fractional_steps"])
def test_pilot_rejects_unverifiable_comparisons(pilot, change):
    hd, h, runs = pilot
    path = prereg.run_dir(h, "real", 1, runs) / "metrics.json"
    m = json.loads(path.read_text())
    if change == "cohort":
        m["final_rows_sha256"]["replay"] = "b" * 64
    elif change == "incomplete":
        m["status"] = "running"
    elif change == "nonfinite":
        m["evals"][-1]["val_nats_per_char_pooled"] = float("nan")
    elif change == "fractional_steps":
        m["steps_done"] = float(m["steps_done"])
    else:
        del m["data_lineage"]
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        report_pilot.summarize(h, runs, hd)
