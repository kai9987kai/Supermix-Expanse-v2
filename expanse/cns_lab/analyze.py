"""Analyse finished Connectome Lab runs: a summary table and, per learned-edge run, what training did to the wiring.

    python expanse/cns_lab/analyze.py                 # every finished run
    python expanse/cns_lab/analyze.py --run <key>     # one run

Summary (``checkpoints/cns_lab/analysis/summary.md`` + ``.json``): every run's
final metrics grouped by hypothesis and arm, the paired deltas and any issued
verdict.

Per run with a saved core (``analysis/<key>.md`` + ``.json``, and a ``.png``
histogram when matplotlib is installed):

* drift ``R_e = W_trained / W_bio`` -- quantiles and a text histogram of log2 R
* R by pathway: node roles input (sensory / ascending / visual projection),
  central, output (descending / motor / efferent / endocrine) for pre and post
* R by (pre superclass -> post superclass), the most changed groups
* the most strengthened / weakened edges (type names, synapse counts)
* hubs: largest changes of weighted in- and out-strength (sum |W|)
* modularity of |W| by superclass before and after (Newman Q, undirected)
* sign flips (0 by construction for ``dale_softplus``)
* leak by superclass, gate usage by source, state autocorrelation, retention
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LAB))

import connectome_full as cf  # noqa: E402
import connectome_lab as cl  # noqa: E402
import prereg  # noqa: E402

AN = prereg.OUT / "analysis"
ROLE_NAMES = {0: "central", 1: "input", 2: "output"}


def modularity(src: np.ndarray, dst: np.ndarray, w: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity of the undirected weighted graph |w| under a node partition."""
    w = np.abs(w).astype(np.float64)
    two_m = 2.0 * w.sum()
    if two_m == 0:
        return float("nan")
    _, lab = np.unique(labels, return_inverse=True)
    k = np.bincount(src, weights=w, minlength=len(labels)) + np.bincount(dst, weights=w, minlength=len(labels))
    inside = np.bincount(lab[src][lab[src] == lab[dst]], weights=2 * w[lab[src] == lab[dst]], minlength=lab.max() + 1)
    tot = np.bincount(lab, weights=k, minlength=lab.max() + 1)
    return float((inside / two_m - (tot / two_m) ** 2).sum())


def text_hist(values: np.ndarray, bins: int = 16, width: int = 40) -> List[str]:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-9:
        return [f"all values = {lo:.4f} ({len(values):,})"]
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    top = counts.max()
    return [f"{edges[i]:+8.4f} .. {edges[i + 1]:+8.4f} | {'#' * int(round(width * c / top)):<{width}} {c:,}"
            for i, c in enumerate(counts)]


def group_stats(keys: np.ndarray, logr: np.ndarray, min_n: int = 1) -> List[Dict[str, Any]]:
    uk, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv)
    s = np.bincount(inv, weights=logr)
    s2 = np.bincount(inv, weights=logr ** 2)
    out = []
    for i, k in enumerate(uk):
        if n[i] >= min_n:
            mean = s[i] / n[i]
            out.append({"group": str(k), "n": int(n[i]), "mean_log2_R": float(mean), "R_geo": float(2 ** mean),
                        "std_log2_R": float(math.sqrt(max(0.0, s2[i] / n[i] - mean ** 2)))})
    return out


def analyse_core(key: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = prereg.RUNS / key / "core.pt"
    if not path.exists():
        return None
    core, _ = cl.load_core_file(path, cache_dir=prereg.OUT / "graphs")
    a = cl.export_anatomy(core)
    an = core.anatomy or {}
    names = np.asarray(an.get("node_id") or [str(i) for i in range(core.n)], dtype=object)
    sc = np.asarray(an.get("superclass") or ["?"] * core.n, dtype=object)
    role = np.zeros(core.n, dtype=np.int64)
    role[a["input_nodes"]] = 1
    role[a["output_nodes"]] = 2
    src, dst = a["edge_src"].astype(np.int64), a["edge_dst"].astype(np.int64)
    bio, learned, r = a["biological_weight"], a["learned_weight"], a["learned_multiplier"].astype(np.float64)
    rep: Dict[str, Any] = {"run": key, "arm": metrics.get("arm"), "seed": metrics.get("seed"),
                           "hypothesis": metrics.get("hypothesis"), "config": metrics.get("config"),
                           "n_types": int(core.n), "n_edges": int(core.nnz)}
    rep["leak"] = (metrics.get("core_report") or {}).get("leak_by_superclass")
    ev = (metrics.get("evals") or [{}])[-1]
    rep["gate_by_source"] = ev.get("gate_by_source")
    rep["state_autocorr"] = ev.get("state_autocorr")
    if hasattr(core, "retain_logit"):
        ret = a["retain"]
        rep["retain"] = {"mean": float(ret.mean()), "by_superclass": {s: float(ret[sc == s].mean()) for s in np.unique(sc)}}
    if not core.cfg.learned_edges:
        rep["edges"] = "fixed (no drift)"
        return rep
    logr = np.log2(np.clip(r, 1e-12, None))
    q = np.quantile(r, [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    rep["R_quantiles"] = dict(zip(["min", "p01", "p10", "p25", "p50", "p75", "p90", "p99", "max"], map(float, q)))
    rep["R_mean"] = float(r.mean())
    rep["frac_R_changed_10pct"] = float((np.abs(r - 1) > 0.1).mean())
    rep["sign_flips"] = int((np.sign(learned) != np.sign(bio)).sum())
    rep["log2R_histogram"] = text_hist(logr)
    path_key = np.array([f"{ROLE_NAMES[int(p)]}->{ROLE_NAMES[int(q_)]}" for p, q_ in zip(role[src], role[dst])], dtype=object)
    rep["R_by_pathway"] = sorted(group_stats(path_key, logr), key=lambda g: g["group"])
    pair_key = sc[src].astype(str) + np.array(["->"] * len(src), dtype=object) + sc[dst].astype(str)
    groups = group_stats(pair_key, logr, min_n=max(20, len(src) // 5000))
    rep["R_by_superclass_pair_most_changed"] = sorted(groups, key=lambda g: -abs(g["mean_log2_R"]))[:15]
    syn = a["synapse_count"]
    solid = np.flatnonzero(syn >= max(1, int(np.median(syn[syn > 0])) if (syn > 0).any() else 1))
    order = solid[np.argsort(r[solid])]

    def edge_row(e: int) -> Dict[str, Any]:
        return {"pre": str(names[src[e]]), "post": str(names[dst[e]]), "pre_superclass": str(sc[src[e]]),
                "post_superclass": str(sc[dst[e]]), "synapses": int(syn[e]), "bio": float(bio[e]),
                "learned": float(learned[e]), "R": float(r[e])}

    rep["most_strengthened"] = [edge_row(int(e)) for e in order[::-1][:12]]
    rep["most_weakened"] = [edge_row(int(e)) for e in order[:12]]
    n = core.n
    d_in = np.bincount(dst, weights=np.abs(learned), minlength=n) - np.bincount(dst, weights=np.abs(bio), minlength=n)
    d_out = np.bincount(src, weights=np.abs(learned), minlength=n) - np.bincount(src, weights=np.abs(bio), minlength=n)

    def hubs(delta: np.ndarray) -> Dict[str, Any]:
        o = np.argsort(delta)
        pick = lambda idx: [{"type": str(names[i]), "superclass": str(sc[i]), "delta_abs_strength": float(delta[i])} for i in idx]  # noqa: E731
        return {"gained": pick(o[::-1][:8]), "lost": pick(o[:8])}

    rep["hubs_in_strength"] = hubs(d_in)
    rep["hubs_out_strength"] = hubs(d_out)
    rep["modularity_by_superclass"] = {"bio": modularity(src, dst, bio, sc.astype(str)),
                                       "learned": modularity(src, dst, learned, sc.astype(str))}
    rep["modularity_by_role"] = {"bio": modularity(src, dst, bio, role), "learned": modularity(src, dst, learned, role)}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(logr, bins=80, color="#3b6ea8")
        ax.set_yscale("log")
        ax.set_xlabel("log2(W_trained / W_bio)")
        ax.set_ylabel("edges")
        ax.set_title(f"{metrics.get('arm')} seed {metrics.get('seed')}: edge drift")
        fig.tight_layout()
        fig.savefig(AN / f"{key}_drift.png", dpi=110)
        plt.close(fig)
        rep["png"] = f"{key}_drift.png"
    except Exception:
        pass
    return rep


def run_markdown(rep: Dict[str, Any]) -> str:
    L = [f"# Lab run {rep['run']} — {rep.get('hypothesis')} / {rep.get('arm')} / seed {rep.get('seed')}", "",
         f"{rep['n_types']:,} types, {rep['n_edges']:,} edges. Config: `{json.dumps(rep.get('config'))}`", ""]
    if "R_quantiles" in rep:
        L += ["## Edge drift R = W_trained / W_bio", "",
              "| " + " | ".join(rep["R_quantiles"]) + " |", "|" + "---|" * len(rep["R_quantiles"]),
              "| " + " | ".join(f"{v:.4f}" for v in rep["R_quantiles"].values()) + " |", "",
              f"mean R {rep['R_mean']:.4f}; edges changed > 10%: {100 * rep['frac_R_changed_10pct']:.2f}%; "
              f"sign flips: {rep['sign_flips']}", "", "log2 R histogram:", "", "```", *rep["log2R_histogram"], "```", "",
              "### By pathway (pre role -> post role)", "", "| pathway | edges | geometric-mean R | std log2 R |",
              "|---|---|---|---|"]
        L += [f"| {g['group']} | {g['n']:,} | {g['R_geo']:.4f} | {g['std_log2_R']:.4f} |" for g in rep["R_by_pathway"]]
        L += ["", "### Most changed superclass pairs", "", "| pre -> post | edges | geometric-mean R |", "|---|---|---|"]
        L += [f"| {g['group']} | {g['n']:,} | {g['R_geo']:.4f} |" for g in rep["R_by_superclass_pair_most_changed"]]
        for title, key in (("Most strengthened edges", "most_strengthened"), ("Most weakened edges", "most_weakened")):
            L += ["", f"### {title} (synapse count >= median)", "", "| pre | post | synapses | W_bio | W_trained | R |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {e['pre']} ({e['pre_superclass']}) | {e['post']} ({e['post_superclass']}) | {e['synapses']} | "
                  f"{e['bio']:+.5f} | {e['learned']:+.5f} | {e['R']:.3f} |" for e in rep[key]]
        L += ["", "### Hubs (change of weighted strength, sum |W|)", ""]
        for side in ("hubs_in_strength", "hubs_out_strength"):
            L.append(f"* {side}: gained " + ", ".join(f"{h['type']} ({h['delta_abs_strength']:+.4f})" for h in rep[side]["gained"][:5])
                     + "; lost " + ", ".join(f"{h['type']} ({h['delta_abs_strength']:+.4f})" for h in rep[side]["lost"][:5]))
        m1, m2 = rep["modularity_by_superclass"], rep["modularity_by_role"]
        L += ["", f"Modularity of |W| by superclass: bio {m1['bio']:.4f} -> trained {m1['learned']:.4f}; "
                  f"by role: bio {m2['bio']:.4f} -> trained {m2['learned']:.4f}"]
    else:
        L.append(f"Edges: {rep.get('edges')}")
    if rep.get("leak"):
        L += ["", "## Leak (integration rate a = sigmoid(leak)) by superclass", "", "| superclass | types | mean a | std |",
              "|---|---|---|---|"]
        L += [f"| {s} | {v['n']} | {v['mean']:.4f} | {v['std']:.4f} |" for s, v in rep["leak"].items()]
    if rep.get("gate_by_source"):
        L += ["", "## Gate usage by source (mean |g| over tokens)", ""]
        L += [f"* {s}: {v:.5f}" for s, v in rep["gate_by_source"].items()]
    if rep.get("state_autocorr") is not None:
        L += ["", f"State autocorrelation (cosine of consecutive token states): {rep['state_autocorr']:.4f}"]
    if rep.get("retain"):
        L += ["", f"Learned retention between tokens: mean {rep['retain']['mean']:.4f}"]
    return "\n".join(L) + "\n"


def summary(rows: List[Dict[str, Any]]) -> str:
    L = ["# Connectome Lab — run summary", "",
         "Final eval of every finished run (nats per reply character; lower is better).", "",
         "| hypothesis | arm | seed | pooled | replay | code | bio | connectome | fresh | temporal | temporal EM | |gate| | s/step |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, p=4):
        return "" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))

    for m in sorted(rows, key=lambda m: (m.get("hypothesis", ""), m.get("arm", ""), m.get("seed", 0))):
        ev = (m.get("evals") or [{}])[-1]
        L.append(f"| {m.get('hypothesis')} | {m.get('arm')} | {m.get('seed')} | {f(ev.get('val_nats_per_char_pooled'))} | "
                 + " | ".join(f(ev.get(f"val_{s}_nats_per_char")) for s in ("replay", "code", "bio", "connectome", "fresh"))
                 + f" | {f(ev.get('temporal_nats_per_char'))} | {f(ev.get('temporal_exact_match'), 3)} | "
                 f"{f(ev.get('gate_abs_mean'))} | {f((m.get('throughput') or {}).get('s_per_step_mean'), 2)} |")
    for vp in sorted(prereg.VERDICTS.glob("*.verdict.json")):
        v = json.loads(vp.read_text(encoding="utf-8"))
        L += ["", f"## {v['id']}: **{v['verdict']}**", "", v["claim"], "",
              f"Primary metric `{v['primary_metric']}` ({v['direction']}); rule {json.dumps(v['success_rule'])}; "
              f"mean delta {v.get('mean_delta', float('nan')):.5f} vs threshold {v.get('threshold', float('nan')):.5f}", "",
              "| seed | control | treatment | delta (+ favours treatment) |", "|---|---|---|---|"]
        L += [f"| {r['seed']} | {f(r.get('control'), 5)} | {f(r.get('treatment'), 5)} | {f(r.get('delta'), 5)} |"
              for r in v["per_seed"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=None)
    args = ap.parse_args(argv)
    AN.mkdir(parents=True, exist_ok=True)
    rows = []
    for mp in sorted(prereg.RUNS.glob("*/metrics.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("status") != "done" or mp.parent.name.endswith("_smoke"):
            continue
        rows.append(m)
        key = mp.parent.name
        if args.run and key != args.run:
            continue
        out_json = AN / f"{key}.json"
        if out_json.exists() and not args.run:
            continue
        rep = analyse_core(key, m)
        if rep is None:
            continue
        cf.write_json(out_json, rep)
        (AN / f"{key}.md").write_text(run_markdown(rep), encoding="utf-8")
        print(f"[analyze] {key}: {m.get('arm')} seed {m.get('seed')}", flush=True)
    (AN / "summary.md").write_text(summary(rows), encoding="utf-8")
    cf.write_json(AN / "summary.json", {"runs": [{k: m.get(k) for k in ("hypothesis", "arm", "seed", "run_key", "config",
                                                                       "throughput", "peak_rss_mb", "wall_minutes")}
                                                  | {"final": (m.get("evals") or [{}])[-1]} for m in rows]})
    print(f"[analyze] {len(rows)} finished runs -> {AN / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
