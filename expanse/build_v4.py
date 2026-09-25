"""Build Supermix Expanse v4: v3 + an input-dependent gate on its male-CNS core.

    python expanse/build_v4.py --steps 600 --threads 8                         # v4 (dynamic gate) -> checkpoint
    python expanse/build_v4.py --steps 600 --gate static --no_save --tag ctrl  # matched control (no checkpoint)

v3's trained ``FullConnectomeCore`` is converted with ``connectome_lab.lab_from_full`` (bitwise the same
function; checked on real rows before training), the dynamic gate term ``tanh(W_g RMSNorm(h))`` is added with
``W_g = 0`` (still exactly v3 at birth), and only the core trains -- the trunk stays frozen, as in the
Connectome Lab, so the change is attributable to the core. Edges stay fixed (they did not learn at this scale
in the lab). The control run trains the same core the same way with the static gate, so ``v4 - control``
isolates the gate from the extra core training. Both are scored on the same v3 dev rows (never trained on)
at step 0 and after training; v4 is saved as a normal Expanse checkpoint (the lab core's config and anatomy
travel with it) plus a receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cns_lab"))

import connectome_full as cf  # noqa: E402
import connectome_lab as cl  # noqa: E402
import expanse_core as ec  # noqa: E402
import run_lab as rl  # noqa: E402
from expanse_core import text_utils  # noqa: E402

CKPT = ROOT / "checkpoints"


def log(*a) -> None:
    print("[v4]", *a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inp", default=str(CKPT / "supermix_expanse_v3.pt"))
    ap.add_argument("--out", default=str(CKPT / "supermix_expanse_v4.pt"))
    ap.add_argument("--gate", choices=("dynamic", "static"), default="dynamic")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr_new", type=float, default=1e-3, help="the new dynamic-gate parameters")
    ap.add_argument("--lr_core", type=float, default=1e-4, help="v3's already-trained core parameters")
    ap.add_argument("--eval_per_source", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=0, help="extra evals every N steps (0: start and end only)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--tag", default="v4")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    model, tok, payload = ec.load_expanse(args.inp)
    payload["state_dict"] = None
    full = model.cns_full
    if not isinstance(full, cf.FullConnectomeCore):
        raise SystemExit(f"{args.inp}: expected a v3 FullConnectomeCore, got {type(full).__name__}")
    base = cf.build_full_graph(cl.npz_path())
    graph = cl.lab_graph(base, cl.LabConfig(), 0)
    cfg = cl.LabConfig(gate=args.gate, edge_param="fixed")
    core = cl.lab_from_full(full, cfg, graph=graph)

    # data: the v3 corpus (same split seed as v3, so its dev rows were never trained on)
    P = {"seq": int(model.config.max_position_embeddings), "eval_per_source": args.eval_per_source,
         "corpus": {"seed": 2026, "dev_frac": 0.05, "dev_cap": 200, "max_rows_per_source": 6000, "fly_rows": 0},
         "temporal": {"train_rows": 0, "eval_rows": 0}}
    data = rl.build_data(model, tok, P, args.seed)
    feats = rl.Features(model, payload)
    log(f"data {data['counts']}")

    # conversion must be exact on real rows before anything trains
    block = next(iter(data["eval"].values()))
    xb = block["x"][:4]
    xb = xb[:, :int((xb != text_utils.PAD).sum(1).max())]
    kw = feats([r["user"] for r in block["rows"][:4]])
    with torch.no_grad():
        ref = model(xb, return_mtp=False, **kw).logits
        ec.attach_lab_core(model, core)
        got = model(xb, return_mtp=False, **kw).logits
    exact = bool(torch.equal(ref, got))
    log(f"lab_from_full conversion exact on real rows: {exact} (max |dlogit| {float((ref - got).abs().max()):.2e})")
    if not exact:
        raise SystemExit("conversion is not exact; refusing to train")
    del full

    # frozen trunk; the core trains (new gate parameters faster than v3's trained ones)
    model.requires_grad_(False)
    core.requires_grad_(True)
    new = [p for n, p in core.named_parameters() if n.startswith(("gate_proj", "gate_norm"))]
    new_ids = {id(p) for p in new}
    old = [p for p in core.parameters() if id(p) not in new_ids]
    groups = [{"params": old, "lr": args.lr_core, "weight_decay": 0.0}]
    if new:
        groups.append({"params": new, "lr": args.lr_new, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in groups]
    warm = max(1, int(0.05 * args.steps))

    def lr_scale(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, args.steps - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    plan = rl.batch_plan(len(data["m_tr"]), 0, args.steps, args.batch, 0, args.seed)
    evals = [rl.evaluate(model, tok, data, feats, args.batch, 0)]
    log(f"[eval 0] pooled {evals[0]['val_nats_per_char_pooled']:.5f} (= v3)")
    params = [p for g in groups for p in g["params"]]
    times = []
    model.eval()
    for step in range(args.steps):
        idx = plan[step]["corpus"]
        xb, yb = data["x_tr"][idx], data["y_tr"][idx]
        L = int((xb != text_utils.PAD).sum(1).max())
        xb, yb = xb[:, :L], yb[:, :L]
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * lr_scale(step)
        t1 = time.perf_counter()
        with torch.enable_grad():
            lg = model(xb, return_mtp=False, **feats([data["m_tr"][int(i)]["user"] for i in idx])).logits
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), yb[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        times.append(time.perf_counter() - t1)
        lv = float(loss.detach())
        if not math.isfinite(lv):
            raise SystemExit(f"non-finite loss at step {step + 1}")
        if (step + 1) % 20 == 0 or step == 0:
            rec = times[-20:]
            log(f"[step {step + 1}/{args.steps}] loss {lv:.4f} |gate| {float(core.gate.detach().abs().mean()):.4f} "
                f"{sum(rec) / len(rec):.2f}s/step eta {(args.steps - step - 1) * sum(rec) / len(rec) / 60:.1f}m")
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            evals.append(rl.evaluate(model, tok, data, feats, args.batch, step + 1))
            log(f"[eval {step + 1}] pooled {evals[-1]['val_nats_per_char_pooled']:.5f}")
    evals.append(rl.evaluate(model, tok, data, feats, args.batch, args.steps))
    first, last = evals[0], evals[-1]
    log(f"[eval {args.steps}] pooled {last['val_nats_per_char_pooled']:.5f} "
        f"(delta vs v3 {last['val_nats_per_char_pooled'] - first['val_nats_per_char_pooled']:+.5f})")
    model.eval()
    report = {
        "tag": args.tag, "gate": args.gate, "steps": args.steps, "batch": args.batch, "lr_new": args.lr_new,
        "lr_core": args.lr_core, "seed": args.seed, "train_scope": "core (trunk frozen)", "edges": "fixed",
        "conversion_exact": exact, "evals": evals, "core_report": rl.core_report(core),
        "s_per_step_mean": sum(times) / max(1, len(times)), "wall_minutes": round((time.time() - t0) / 60, 1),
        "delta_pooled_vs_v3": last["val_nats_per_char_pooled"] - first["val_nats_per_char_pooled"],
        "delta_by_source_vs_v3": {k: last[k] - first[k] for k in last if k.startswith("val_") and k.endswith("_per_char")},
    }
    rpath = CKPT / f"build_{args.tag}.json"
    cf.write_json(rpath, report)
    log(f"report -> {rpath}")
    if not args.no_save:
        receipt = dict(payload.get("expanse") or {})
        v4_info = {k: v for k, v in report.items() if k != "evals"}
        v4_info["evals_pooled"] = [e["val_nats_per_char_pooled"] for e in evals]
        v4_info["note"] = ("receipt['training'] is the BASE model's (v3) training; v4 adds core-only steps on top. "
                           "The static-gate control trained its gate at lr_core while v4's new gate parameters used "
                           "lr_new, so 'v4 - control' is an upper bound on the input-dependent share of the gain.")
        receipt.update({"stage": "v4-dynamic-cns-gate", "v4": v4_info, "v4_evals": evals,
                        "base_checkpoint": str(Path(args.inp).name), "base_sha256": ec.sha256_file(args.inp)})
        extra = dict(payload.get("extra") or {})
        extra.update({"note": "v4: v3 + input-dependent gate on the male-CNS core (core-only training)",
                      "steps": args.steps})
        ec.save_expanse(args.out, model, tok, extra, receipt)
        Path(args.out).with_suffix(".receipt.json").write_text(
            json.dumps(ec.jsonable({k: v for k, v in receipt.items() if k not in ("omni7_meta", "cns")}), indent=1),
            encoding="utf-8")
        log(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
