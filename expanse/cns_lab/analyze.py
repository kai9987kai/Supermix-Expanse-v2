"""Analyse finished Connectome Lab runs: a summary table and, per learned-edge run, what training did to the wiring.

    python expanse/cns_lab/analyze.py                 # every finished run
    python expanse/cns_lab/analyze.py --run <key>     # one run

Summary (``checkpoints/cns_lab/analysis/summary.md`` + ``.json``): every run's
final metrics grouped by hypothesis and arm, the paired deltas and any issued
verdict; the final-eval ablations / core-write diagnostics of the runs that
recorded them; and, per exploratory hypothesis (``exploratory: true``), an
"Exploratory arms" section: every arm's final pooled / per-source / temporal
numbers and its delta against the file's reference arm (``reset`` if present,
else the first arm), flagged against a noise floor -- ``--noise_floor`` if
given, else per metric the SD over seeds of the paired treatment - control
delta of the pre-registered hypotheses run under the same protocol (for the
3,000-type subgraph protocol: H-CNS-003's persist - reset). Single seed, no
claims.

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
from typing import Any, Dict, List, Optional, Tuple

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


#: Final-eval numbers of the exploratory table (key, column); lower is better for all of them.
EXPLORATORY_METRICS = (("val_nats_per_char_pooled", "pooled"), ("val_replay_nats_per_char", "replay"),
                       ("val_code_nats_per_char", "code"), ("val_bio_nats_per_char", "bio"),
                       ("val_connectome_nats_per_char", "connectome"), ("val_fresh_nats_per_char", "fresh"),
                       ("temporal_nats_per_char", "temporal"), ("temporal_answer_nats_per_char", "temporal answer"))
#: Ablation / diagnostic keys run_lab adds when a protocol opts in (or, for the write stats, when a core exists).
DIAGNOSTIC_METRICS = (("val_nats_per_char_pooled_zero_edges", "pooled, edges zeroed"),
                      ("val_nats_per_char_pooled_gate0", "pooled, gate closed"),
                      ("graph_dependent_gain", "graph-dependent gain"), ("core_gain", "core gain"),
                      ("write_rms_ratio", "write RMS / hidden RMS"), ("write_token_var_frac", "write token-var frac"))
NOISE_K = 2.0


def final_eval(m: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return ((m or {}).get("evals") or [{}])[-1]


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) else None


def _read_metrics(h: Dict[str, Any], arm: str, seed: int, runs: Path) -> Optional[Dict[str, Any]]:
    mp = prereg.run_dir(h, arm, seed, runs) / "metrics.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _protocol_id(p: Dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in p.items() if k != "notes"}, sort_keys=True)


def locked_hypotheses(hyp_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for hid in prereg.all_hypotheses(hyp_dir):
        try:
            out.append(prereg.load_locked(hid, hyp_dir))
        except (prereg.PreregError, ValueError, KeyError):
            continue
    return out


def paired_noise_floor(hyps: List[Dict[str, Any]], protocol: Dict[str, Any], runs: Path,
                       metrics: Tuple[str, ...]) -> Tuple[Dict[str, float], List[str]]:
    """Per metric: the SD (ddof 1) over seeds of the paired treatment - control delta of every finished
    pre-registered hypothesis run under ``protocol`` (the largest if several). Also returns their ids."""
    floors: Dict[str, float] = {}
    used: List[str] = []
    pid = _protocol_id(protocol)
    for h in hyps:
        if h.get("exploratory") or _protocol_id(h["protocol"]) != pid:
            continue
        pairs = []
        for s in h["seeds"]:
            c, t = (_read_metrics(h, h[f"{r}_arm"], s, runs) for r in ("control", "treatment"))
            if c and t and c.get("status") == "done" and t.get("status") == "done":
                pairs.append((final_eval(c), final_eval(t)))
        if len(pairs) < 2:
            continue
        used.append(h["id"])
        for key in metrics:
            d = [_num(t.get(key)) - _num(c.get(key)) for c, t in pairs
                 if _num(t.get(key)) is not None and _num(c.get(key)) is not None]
            if len(d) >= 2:
                floors[key] = max(floors.get(key, 0.0), float(np.std(d, ddof=1)))
    return floors, used


def exploratory_section(hyp_dir: Optional[Path] = None, runs: Optional[Path] = None,
                        noise_floor: Any = None, noise_k: float = NOISE_K) -> List[str]:
    """Markdown lines: one block per exploratory hypothesis (see the module doc)."""
    hyp_dir = Path(hyp_dir) if hyp_dir is not None else prereg.HYP_DIR
    runs = Path(runs) if runs is not None else prereg.RUNS
    hyps = locked_hypotheses(hyp_dir)
    keys = tuple(k for k, _ in EXPLORATORY_METRICS)
    L: List[str] = []

    def f(v, p=5, signed=False):
        v = _num(v)
        return "" if v is None else (f"{v:+.{p}f}" if signed else f"{v:.{p}f}")

    for h in hyps:
        if not h.get("exploratory"):
            continue
        arms = list(h["arms"])
        ref = "reset" if "reset" in h["arms"] else arms[0]
        if isinstance(noise_floor, dict):
            floors, source = {k: float(v) for k, v in noise_floor.items()}, "passed in (per metric)"
        elif noise_floor is not None:
            floors, source = {k: float(noise_floor) for k in keys}, f"passed in ({float(noise_floor):.5f} for every metric)"
        else:
            floors, used = paired_noise_floor(hyps, h["protocol"], runs, keys)
            source = (f"SD over seeds of the paired treatment - control delta of {', '.join(used)} (same protocol)"
                      if used else "none available (no finished pre-registered runs under this protocol)")
        L += ["", f"## Exploratory arms: {h['id']} — single seed, no claims", "",
              f"> **Single seed, no claims.** {h['claim']} Exploratory arms are reported, never given a verdict; "
              f"a flag only says a difference is larger than the seed-to-seed spread of a paired comparison.", "",
              f"Reference arm: `{ref}`. Delta = arm - `{ref}` in nats per reply character at the final eval "
              f"(negative = lower loss than the reference). `*` = |delta| > {noise_k:g} x noise floor; "
              f"`?` = no noise floor for that metric. Noise floor: {source}.", ""]
        if floors:
            L += ["| metric | noise floor |", "|---|---|"]
            L += [f"| {name} | {floors[k]:.5f} |" for k, name in EXPLORATORY_METRICS if k in floors]
            L.append("")
        L += ["| arm | seed | status | " + " | ".join(n for _, n in EXPLORATORY_METRICS) + " |",
              "|---|---|---|" + "---|" * len(EXPLORATORY_METRICS)]
        deltas: List[str] = []
        diag: List[str] = []
        def done_eval(m):  # only a finished run has final numbers
            return final_eval(m) if (m or {}).get("status") == "done" else {}

        for s in h["seeds"]:
            ref_ev = done_eval(_read_metrics(h, ref, s, runs))
            for arm in arms:
                m = _read_metrics(h, arm, s, runs)
                ev = done_eval(m)
                st = (m or {}).get("status", "todo")
                if m is not None and st != "done" and (m.get("evals") or []):
                    st += f" (at step {m['evals'][-1].get('step')})"
                L.append(f"| {arm} | {s} | {st} | " + " | ".join(f(ev.get(k)) for k in keys) + " |")
                if arm != ref:
                    cells = []
                    for k in keys:
                        a, b = _num(ev.get(k)), _num(ref_ev.get(k))
                        if a is None or b is None:
                            cells.append("")
                            continue
                        d = a - b
                        fl = floors.get(k)
                        mark = "?" if fl is None else ("*" if abs(d) > noise_k * fl else "")
                        cells.append(f"{d:+.5f}{mark}")
                    deltas.append(f"| {arm} | {s} | " + " | ".join(cells) + " |")
                if any(_num(ev.get(k)) is not None for k, _ in DIAGNOSTIC_METRICS):
                    diag.append(f"| {arm} | {s} | " + " | ".join(f(ev.get(k)) for k, _ in DIAGNOSTIC_METRICS) + " |")
        L += ["", f"Delta vs `{ref}`:", "",
              "| arm | seed | " + " | ".join(n for _, n in EXPLORATORY_METRICS) + " |",
              "|---|---|" + "---|" * len(EXPLORATORY_METRICS), *deltas]
        if diag:
            L += ["", "Final-eval ablations and core-write diagnostics:", "",
                  "| arm | seed | " + " | ".join(n for _, n in DIAGNOSTIC_METRICS) + " |",
                  "|---|---|" + "---|" * len(DIAGNOSTIC_METRICS), *diag]
    return L


def diagnostics_section(rows: List[Dict[str, Any]]) -> List[str]:
    """Markdown lines: the ablation / diagnostic metrics of every run that recorded any."""
    out = []
    for m in sorted(rows, key=lambda m: (m.get("hypothesis", ""), m.get("arm", ""), m.get("seed", 0))):
        ev = final_eval(m)
        if any(_num(ev.get(k)) is not None for k, _ in DIAGNOSTIC_METRICS):
            out.append(f"| {m.get('hypothesis')} | {m.get('arm')} | {m.get('seed')} | "
                       + " | ".join("" if _num(ev.get(k)) is None else f"{_num(ev.get(k)):.5f}" for k, _ in DIAGNOSTIC_METRICS)
                       + " |")
    if not out:
        return []
    return ["", "## Final-eval ablations and core-write diagnostics", "",
            "Pooled nats/char with every edge weight zeroed (no signal crosses the graph: a constant core write "
            "only with a static gate and state=reset; a dynamic gate still scales it per token, so for "
            "dynamic-gate arms the graph-dependent gain excludes what the gate alone can do) and with the gate "
            "closed; graph-dependent gain = zeroed - trained, core gain = gate closed - trained (the gain of "
            "the core's write given the trained trunk); write RMS relative to the hidden state and the share "
            "of the write's power that varies across tokens, both over every eval token.", "",
            "| hypothesis | arm | seed | " + " | ".join(n for _, n in DIAGNOSTIC_METRICS) + " |",
            "|---|---|---|" + "---|" * len(DIAGNOSTIC_METRICS), *out]


def summary(rows: List[Dict[str, Any]], hyp_dir: Optional[Path] = None, runs: Optional[Path] = None,
            noise_floor: Any = None, noise_k: float = NOISE_K) -> str:
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
    L += diagnostics_section(rows)
    L += exploratory_section(hyp_dir, runs, noise_floor, noise_k)
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=None)
    ap.add_argument("--noise_floor", default=None,
                    help="exploratory flags: a float for every metric or a JSON {metric: floor} "
                         "(default: derived from the pre-registered paired runs of the same protocol)")
    ap.add_argument("--noise_k", type=float, default=NOISE_K, help="flag |delta| > k x noise floor")
    args = ap.parse_args(argv)
    floor: Any = None
    if args.noise_floor is not None:
        floor = json.loads(args.noise_floor) if args.noise_floor.strip().startswith("{") else float(args.noise_floor)
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
    (AN / "summary.md").write_text(summary(rows, noise_floor=floor, noise_k=args.noise_k), encoding="utf-8")
    cf.write_json(AN / "summary.json", {"runs": [{k: m.get(k) for k in ("hypothesis", "arm", "seed", "run_key", "config",
                                                                       "throughput", "peak_rss_mb", "wall_minutes")}
                                                  | {"final": (m.get("evals") or [{}])[-1]} for m in rows]})
    print(f"[analyze] {len(rows)} finished runs -> {AN / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
