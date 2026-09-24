"""Benchmark the Connectome Lab core: backends x graph size x edge learning x state mode.

    python expanse/cns_lab/bench_connectome.py --threads 8

For each combination: seconds per forward and per backward of the core alone
on a (batch x seq) block of hidden states, and the process's peak working set.
Writes ``checkpoints/cns_lab/bench.json`` and ``bench.md``. Combinations that
cannot fit (dense on the full graph) or would take minutes per step are
skipped and listed as such.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT / "src"))

import connectome_full as cf  # noqa: E402
import connectome_lab as cl  # noqa: E402

OUT = ROOT / "checkpoints" / "cns_lab"


def peak_rss_mb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2 ** 20, 1)
    except Exception:  # pragma: no cover
        return float("nan")


def bench_one(graph, cfg: cl.LabConfig, batch: int, seq: int, hidden: int, repeats: int) -> dict:
    core = cl.fresh_lab_core(hidden, graph, cfg, seed=0)
    with torch.no_grad():
        core.gate.fill_(0.05)
    x = torch.randn(batch, seq, hidden, requires_grad=True)
    fwd, bwd = [], []
    for i in range(repeats + 1):  # first = warm-up
        core.zero_grad(set_to_none=True)
        x.grad = None
        t0 = time.perf_counter()
        out, _ = core(x)
        t1 = time.perf_counter()
        out.pow(2).mean().backward()
        t2 = time.perf_counter()
        if i:
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)
    with torch.no_grad():
        t0 = time.perf_counter()
        core(x)
        infer = time.perf_counter() - t0
    del core, x, out
    gc.collect()
    return {"forward_s": round(min(fwd), 3), "backward_s": round(min(bwd), 3),
            "step_s": round(min(fwd) + min(bwd), 3), "inference_s": round(infer, 3), "peak_rss_mb": peak_rss_mb()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--sizes", default="full,3000,1500")
    ap.add_argument("--backends", default="csr_slab,torch_sparse,dense")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=str(OUT / "bench.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    base = cf.build_full_graph(cl.npz_path())
    rows = []
    for size in args.sizes.split(","):
        sub = None if size == "full" else int(size)
        graph = cl.lab_graph(base, cl.LabConfig(subgraph=sub), 0)
        nnz = len(graph["post"])
        print(f"[bench] graph {size}: {graph['n']} types, {nnz:,} edges", flush=True)
        for backend in args.backends.split(","):
            for edge in ("fixed", "dale_softplus"):
                for state in ("reset", "persist"):
                    row = {"graph": size, "n": graph["n"], "nnz": nnz, "backend": backend, "edge_param": edge,
                           "state": state}
                    skip = None
                    if backend == "dense" and graph["n"] > 4000:
                        skip = "dense n x n too large"
                    elif backend != "csr_slab" and state == "persist" and nnz > 1_000_000:
                        skip = "per-token scan on the full graph: csr_slab only"
                    if skip:
                        row["skipped"] = skip
                    else:
                        cfg = cl.LabConfig(subgraph=sub, edge_param=edge, state=state, backend=backend)
                        try:
                            row.update(bench_one(graph, cfg, args.batch, args.seq, args.hidden, args.repeats))
                        except Exception as exc:  # report, do not abort the sweep
                            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    rows.append(row)
                    print("[bench]", json.dumps(row), flush=True)
    res = {"threads": args.threads, "batch": args.batch, "seq": args.seq, "hidden": args.hidden,
           "torch": torch.__version__, "dtype": "float32", "rows": rows,
           "note": "core alone (no trunk); reset runs all B*T tokens as columns, persist scans T positions"}
    out = Path(args.out)
    cf.write_json(out, res)
    lines = ["# Connectome Lab core benchmark", "",
             f"{args.batch} x {args.seq} tokens, hidden {args.hidden}, {args.threads} threads, fp32, CPU "
             f"(torch {torch.__version__}). Core alone; seconds per call.", "",
             "| graph | edges | backend | edges learned | state | forward s | backward s | step s | process peak RSS MB (running max) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tail = (f"{r['forward_s']} | {r['backward_s']} | {r['step_s']} | {r['peak_rss_mb']}" if "step_s" in r
                else f"{r.get('skipped') or r.get('error')} | | | ")
        lines.append(f"| {r['graph']} | {r['nnz']:,} | {r['backend']} | {r['edge_param']} | {r['state']} | {tail} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[bench] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
