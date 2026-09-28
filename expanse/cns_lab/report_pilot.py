"""Validate and summarize an exploratory pilot without issuing a verdict.

    python expanse/cns_lab/report_pilot.py --hypothesis X-CNS-002

Stored evidence is checked, never rewritten. These descriptive differences
are conditional on the recorded checkpoints, rows, seeds and training budget.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import prereg


def summarize(h, runs=prereg.RUNS, hyp_dir=prereg.HYP_DIR):
    if not h.get("exploratory"):
        raise ValueError("report_pilot requires an exploratory hypothesis")
    records, comparisons, inputs = [], [], []
    reference = next(iter(h["arms"]))
    for seed in h["seeds"]:
        accepted = {}
        for arm in h["arms"]:
            directory = prereg.run_dir(h, arm, seed, runs)
            path = directory / "metrics.json"
            raw = path.read_bytes()
            m = json.loads(raw)
            problems = prereg.check_run(h, arm, seed, directory, m, hyp_dir)
            if m.get("status") != "done" or (
                not m.get("untrainable") and prereg._exact_int(m.get("steps_done")) != h["protocol"]["steps"]
            ):
                problems.append("run is not complete")
            for field in ("input_manifest", "data_lineage", "final_rows_sha256"):
                if field not in m:
                    problems.append(f"pilot requires {field}")
            final = m["evals"][-1]
            required = ["val_nats_per_char_pooled"]
            if h["protocol"].get("final_ablations"):
                required += ["graph_dependent_gain", "core_gain",
                             "val_nats_per_char_pooled_zero_edges", "val_nats_per_char_pooled_gate0"]
            for key in required:
                value = final.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    problems.append(f"missing/nonfinite {key}")
            if problems:
                raise ValueError(f"{arm} seed {seed}: " + "; ".join(problems))
            accepted[arm] = m
            inputs.append({"arm": arm, "seed": seed, "run": directory.name,
                           "metrics_sha256": prereg.sha256_bytes(raw)})
            records.append({"arm": arm, "seed": seed, "run": directory.name,
                            "initial_pooled": m["evals"][0].get("val_nats_per_char_pooled"),
                            "final": final, "throughput": m.get("throughput"),
                            "trainable_params": m.get("trainable_params"),
                            "wall_minutes": m.get("wall_minutes")})
        base = accepted[reference]
        for arm, m in accepted.items():
            pairing, bad = prereg._paired_evidence(base, m)
            if bad:
                raise ValueError(f"{arm} seed {seed}: " + "; ".join(bad))
            comparisons.append({"reference": reference, "arm": arm, "seed": seed,
                                "paired_evidence": pairing,
                                "delta_nats_per_char": base["evals"][-1]["val_nats_per_char_pooled"]
                                - m["evals"][-1]["val_nats_per_char_pooled"]})
    means = {}
    for arm in h["arms"]:
        values = [r["final"]["val_nats_per_char_pooled"] for r in records if r["arm"] == arm]
        means[arm] = {"mean_nats_per_char": statistics.mean(values),
                      "seed_sd": statistics.stdev(values) if len(values) > 1 else None}
    return {"schema": "supermix-expanse-cns-pilot-report-v1", "hypothesis": h["id"],
            "hypothesis_sha256": h["_lock"]["sha256"], "status": "exploratory_only",
            "promotion_allowed": False, "inputs": inputs, "records": records,
            "means": means, "comparisons": comparisons,
            "limitations": "Short development pilot; no confirmatory verdict, generation-quality claim, "
                           "or biological-topology superiority claim. Seed SD is descriptive, not a confidence interval."}


def markdown(report):
    lines = [f"# {report['hypothesis']} exploratory pilot", "", report["limitations"], "",
             "All compared arms have matching recorded input, data-order and evaluation-cohort receipts.", "",
             "| Arm | Seed | Final nats/char | Graph-dependent gain | Core gain | Seconds/step |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in report["records"]:
        f = r["final"]
        def fmt(value):
            return "—" if value is None else f"{value:.6f}"
        lines.append(f"| {r['arm']} | {r['seed']} | {fmt(f['val_nats_per_char_pooled'])} | "
                     f"{fmt(f.get('graph_dependent_gain'))} | {fmt(f.get('core_gain'))} | "
                     f"{fmt((r.get('throughput') or {}).get('s_per_step_mean'))} |")
    lines += ["", "Graph-dependent gain = zero-edge loss minus trained loss; core gain = gate-zero loss "
              "minus trained loss. Positive values favor the trained core. Neither is a real-versus-shuffled comparison.",
              "", "| Arm | Mean nats/char | Across-seed SD |", "|---|---:|---:|"]
    for arm, values in report["means"].items():
        lines.append(f"| {arm} | {values['mean_nats_per_char']:.6f} | {fmt(values['seed_sd'])} |")
    lines += ["", "No model was promoted. Full per-source losses, pairing and metric-file hashes are in the adjacent JSON.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--runs", type=Path, default=prereg.RUNS)
    ap.add_argument("--hyp-dir", type=Path, default=prereg.HYP_DIR)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    h = prereg.load_locked(args.hypothesis, args.hyp_dir)
    report = summarize(h, args.runs, args.hyp_dir)
    out = args.output or prereg.OUT / f"{h['id']}.report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
