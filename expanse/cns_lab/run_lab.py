"""One matched Connectome Lab run: (hypothesis, arm, seed) -> metrics.json.

    python expanse/cns_lab/run_lab.py --hypothesis H-CNS-001 --arm real --seed 1 --threads 8
    python expanse/cns_lab/run_lab.py --hypothesis H-CNS-001 --arm real --seed 1 --smoke --unlocked

What is matched across arms (for one seed): the base model (v3, with its CNS
core replaced), the lab core's initialisation of everything but the graph
(``fresh_lab_core(seed)``: gate 0, so every arm starts as the same function),
the training rows and their order, the optimiser, learning rates, schedule,
step count, eval points and eval rows. What differs is only the arm's
``LabConfig`` (graph, edge parametrisation, state mode, ...).

By default only the lab core trains (``train_scope=core``); the trunk runs in
inference mode (eval: deterministic MoE routing) with its gradients off, so an
arm's effect is the core's alone. ``topology=none`` has nothing to train and is
evaluated as-is at every eval point (``untrainable``).

Loss: next-token cross-entropy on reply tokens. Metrics (each eval point, on
fixed rows): nats per reply character per source and pooled over the corpus
sources (``val_nats_per_char_pooled``), the same on held-out temporal rows,
temporal exact match (final eval, greedy decode), gate usage by source, CNS
state autocorrelation, mean rate, leak by superclass, edge-multiplier drift,
seconds per step and peak RSS. Written to
``checkpoints/cns_lab/runs/<run key>/metrics.json`` with the trained core
(``core.pt``). A finished run is skipped; an unfinished one restarts.

Always recorded as well (additive keys; existing keys keep their meaning):

* ``final_rows`` -- per eval row of the final eval, ``[nats, chars, tokens]``
  per source in eval-row order (``chars`` = len(reply) + 1, as in the
  per-char metrics), with ``final_rows_sha256`` (hash of the rows' keys, to
  check that two runs are paired) and, for temporal rows,
  ``final_rows_temporal_answer`` (``[answer nats, len(reply), answer tokens]``);
* per eval: ``temporal_answer_nats_per_char`` (reply tokens *without* the final
  end-of-reply token, per reply character), ``temporal_nats_per_char_by_task``
  and ``temporal_answer_nats_per_char_by_task``; with a core,
  ``write_rms_ratio`` (RMS of the gated write / RMS of the hidden state at the
  graft) and ``write_token_var_frac`` (share of the write's power that varies
  across tokens; ~0 = the core acts as a constant bias).

Optional protocol keys (absent = the original behaviour, bit for bit; a new
key changes the run key, so it never collides with an existing run):

    eps_edge          AdamW eps of the edge-parameter group only (default: AdamW's 1e-8).
                      Per-edge gradients (~1e-9) sit below 1e-8, so edges barely learn.
    eval_sampling     "head" (dev[:n], default) | "hash" (each source's dev pool sorted by
                      sha256(eval_salt + row_key(user, assistant)), then the first n)
    eval_salt         salt of the "hash" sampling (default "cns-lab-eval-v1")
    final_ablations   true: at the final eval also the pooled nats/char with every edge weight
                      zeroed (``..._zero_edges``: a constant write) and with the gate closed
                      (``..._gate0``), ``graph_dependent_gain`` = zero_edges - trained and
                      ``core_gain`` = gate0 - trained; the core is restored bit for bit
    temporal.split_version   1 (default) | 2 (temporal_tasks.split_rows: near-duplicates of
                      held-out rows removed from the temporal train rows)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB))

import connectome_full as cf  # noqa: E402
import connectome_lab as cl  # noqa: E402
import expanse_core as ec  # noqa: E402
import prereg  # noqa: E402
import temporal_tasks as tt  # noqa: E402
import train_expanse as te  # noqa: E402
from expanse_core import PATHS, text_utils  # noqa: E402

METRICS_SCHEMA = "supermix-expanse-cns-lab-run-v1"
DEFAULT_SOURCES = ("replay", "code", "bio", "connectome", "fresh")
EVAL_SAMPLINGS = ("head", "hash")
DEFAULT_EVAL_SALT = "cns-lab-eval-v1"


def log(*a) -> None:
    print("[lab]", *a, flush=True)


def peak_rss_mb() -> Optional[float]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2 ** 20, 1)
    except Exception:  # pragma: no cover
        return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    cf.write_json(path, obj)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def eval_row_hash(row: Dict[str, Any], salt: str = DEFAULT_EVAL_SALT) -> str:
    """Stable, salted hash of one row (the ``eval_sampling="hash"`` order)."""
    return hashlib.sha256(f"{salt}\x1f{ec.row_key(row['user'], row['assistant'])}".encode("utf-8")).hexdigest()


def select_eval_rows(dev: List[Dict[str, Any]], n: int, sampling: str = "head",
                     salt: str = DEFAULT_EVAL_SALT) -> List[Dict[str, Any]]:
    """The ``n`` eval rows of one source's dev pool: ``head`` = ``dev[:n]`` (original behaviour);
    ``hash`` = the pool sorted by :func:`eval_row_hash` (stable; independent of file order), first ``n``."""
    if sampling == "head":
        return dev[:n]
    if sampling == "hash":
        return sorted(dev, key=lambda r: eval_row_hash(r, salt))[:n]
    raise ValueError(f"eval_sampling {sampling!r}; allowed {EVAL_SAMPLINGS}")


def temporal_split(P: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """The protocol's temporal train / held-out rows (``temporal.split_version`` 1 unless set)."""
    tp = P.get("temporal") or {}
    return tt.split_rows(int(tp.get("train_rows", 0)), int(tp.get("eval_rows", 0)), int(tp.get("seed", 7)),
                         version=int(tp.get("split_version", 1)))


def build_data(model, tok, P: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Train pools (corpus + temporal), fixed eval rows. Rows depend only on the protocol; order on the seed."""
    c = P.get("corpus") or {}
    sources = tuple(P.get("sources") or DEFAULT_SOURCES)
    corpus = te.load_corpus(model.fly_core, fly_rows=int(c.get("fly_rows", 0)), seed=int(c.get("seed", 2026)),
                            dev_frac=float(c.get("dev_frac", 0.05)), dev_cap=int(c.get("dev_cap", 200)),
                            max_rows_per_source=int(c.get("max_rows_per_source", 6000)), v3=True)
    seq = int(P.get("seq", 128))
    train = [r for s in sources for r in corpus[s]["train"]]
    n_eval = int(P.get("eval_per_source", 60))
    sampling = str(P.get("eval_sampling", "head"))
    salt = str(P.get("eval_salt", DEFAULT_EVAL_SALT))
    evals = {s: select_eval_rows(corpus[s]["dev"], n_eval, sampling, salt) for s in sources}
    temporal = temporal_split(P)
    x_tr, y_tr, _, m_tr, dropped = te.encode_rows(train, tok, seq)
    x_tt, y_tt, _, m_tt, dropped_t = te.encode_rows(temporal["train"], tok, seq) if temporal["train"] else (None, None, None, [], 0)
    ev = {}
    for s, rows in list(evals.items()) + ([("temporal", temporal["heldout"])] if temporal["heldout"] else []):
        x, y, _, meta, d = te.encode_rows(rows, tok, seq)
        ev[s] = {"x": x, "y": y, "rows": meta, "dropped": d}
    return {"x_tr": x_tr, "y_tr": y_tr, "m_tr": m_tr, "x_tt": x_tt, "y_tt": y_tt, "m_tt": m_tt, "eval": ev,
            "temporal_heldout": temporal["heldout"], "sources": sources,
            "counts": {"train": len(m_tr), "temporal_train": len(m_tt), "dropped": dropped + dropped_t,
                       "eval": {s: len(v["rows"]) for s, v in ev.items()}}}


def batch_plan(n_corpus: int, n_temporal: int, steps: int, batch: int, temporal_per_batch: int, seed: int) -> List[Dict[str, torch.Tensor]]:
    """Row indices for every step: ``batch - temporal_per_batch`` corpus rows + temporal rows, seed-ordered."""
    g = torch.Generator().manual_seed(int(seed))
    n_c = batch - (temporal_per_batch if n_temporal else 0)

    def stream(n: int, k: int) -> List[torch.Tensor]:
        out, perm, ptr = [], torch.randperm(n, generator=g), 0
        for _ in range(steps):
            if ptr + k > n:
                perm, ptr = torch.randperm(n, generator=g), 0
            out.append(perm[ptr:ptr + k])
            ptr += k
        return out

    corp = stream(n_corpus, n_c)
    temp = stream(n_temporal, temporal_per_batch) if n_temporal and temporal_per_batch else [None] * steps
    return [{"corpus": a, "temporal": b} for a, b in zip(corp, temp)]


class Features:
    """omni v48/v38 features and omni7 states per prompt (the frozen grafts' inputs), cached."""

    def __init__(self, model, payload):
        self.model = model
        self.o7: Dict[str, torch.Tensor] = {}
        if model.omni7 is not None:
            sha = str((((payload.get("expanse") or {}).get("grafts") or {}).get("omni7") or {}).get("sha256", "unknown"))
            cpath = PATHS["data"] / "omni7_states.pt"
            if cpath.exists():
                c = torch.load(cpath, map_location="cpu", weights_only=False)
                if c.get("version") == te.O7_CACHE_VERSION and c.get("v7_sha256") == sha:
                    self.o7 = dict(zip(c["keys"], c["state"]))

    def __call__(self, users: List[str]) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"omni_features": ec.arch.OmniCore.featurize(users)}
        if self.model.omni7 is not None:
            need = [u for u in dict.fromkeys(users) if ec.row_key(u, "") not in self.o7]
            if need:
                with torch.no_grad():
                    st = self.model.omni7_state_for(need)
                for u, s in zip(need, st):
                    self.o7[ec.row_key(u, "")] = s.clone()
            kw["omni7_state"] = torch.stack([self.o7[ec.row_key(u, "")] for u in users])
        return kw


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def _eval_batch(model, feats, x: torch.Tensor, y: torch.Tensor, rows: List[Dict[str, Any]], i: int, batch: int):
    """One eval batch's forward: ``(yb, ce, m)`` with ``ce`` (B, L-1) per-position nats and ``m`` the reply mask."""
    xb, yb = x[i:i + batch], y[i:i + batch]
    L = int((xb != text_utils.PAD).sum(1).max())  # right-padded: trimming PAD columns is exact
    xb, yb = xb[:, :L], yb[:, :L]
    users = [r["user"] for r in rows[i:i + batch]]
    lg = model(xb, return_mtp=False, **feats(users)).logits
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), yb[:, 1:].reshape(-1),
                         reduction="none").view(xb.shape[0], -1)
    m = yb[:, 1:] != -100
    return yb, ce, m


def answer_mask(yb: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``m`` without each row's final end-of-reply target (the answer tokens only)."""
    idx = torch.arange(m.shape[1], device=m.device).unsqueeze(0)
    last = torch.where(m, idx, torch.full_like(idx, -1)).max(dim=1).values
    tgt = yb[:, 1:]
    last_is_eos = tgt.gather(1, last.clamp_min(0).unsqueeze(1)).squeeze(1) == text_utils.EOS
    return m & ~((idx == last.unsqueeze(1)) & (last_is_eos & (last >= 0)).unsqueeze(1))


def _weighted_mean(pairs: List[Any]) -> Optional[float]:
    """Token-weighted mean of ``(weight, value)`` pairs (None when there are none)."""
    tot = sum(w for w, _ in pairs)
    return float(sum(w * v for w, v in pairs) / tot) if pairs and tot > 0 else None


@torch.no_grad()
def evaluate(model, tok, data, feats: Features, batch: int, step: int, exact_match_items: int = 0,
             row_losses: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Losses and core statistics on the fixed eval rows. ``row_losses`` (a dict) receives the per-row
    losses: ``rows`` {source: [[nats, chars, tokens], ...]}, ``keys_sha256`` {source: hash of the
    rows' keys} and ``temporal_answer`` [[answer nats, len(reply), answer tokens], ...]."""
    model.eval()
    rec: Dict[str, Any] = {"step": int(step)}
    pooled = [0.0, 0, 0]  # nats, chars, tokens over the corpus sources
    gate_by_src: Dict[str, List[float]] = {}
    autocorr: List[float] = []
    rates: List[float] = []
    active: List[float] = []
    write_ratio: List[Any] = []  # (tokens, value) per forward
    write_var: List[Any] = []
    per_row: Dict[str, List[List[float]]] = {}
    answer_rows: List[List[float]] = []
    by_task: Dict[str, List[float]] = {}  # task -> [nats, chars, answer nats, answer chars]
    t0 = time.time()
    for src, block in data["eval"].items():
        nats = chars = toks = 0
        x, y, rows = block["x"], block["y"], block["rows"]
        for i in range(0, x.shape[0], batch):
            yb, ce, m = _eval_batch(model, feats, x, y, rows, i, batch)
            nats += float(ce[m].sum())
            toks += int(m.sum())
            chars += sum(len(r["assistant"]) + 1 for r in rows[i:i + batch])  # +1: end of reply
            info = (model.last_graft_info or {}).get("cns_full") or {}
            if info.get("gate_row_mean"):
                gate_by_src.setdefault(src, []).extend(info["gate_row_mean"])
            if info.get("state_autocorr") is not None:
                autocorr.append(float(info["state_autocorr"]))
            if "mean_rate" in info:
                rates.append(info["mean_rate"])
                active.append(info["fraction_active"])
            if info.get("write_rms_ratio") is not None:
                write_ratio.append((int(info.get("M", 1)), float(info["write_rms_ratio"])))
            if info.get("write_token_var_frac") is not None:
                write_var.append((int(info.get("M", 1)), float(info["write_token_var_frac"])))
            # additive per-row / per-task accounting (the metrics above are computed exactly as before)
            ce0 = ce.masked_fill(~m, 0.0).double()
            row_nats, row_toks = ce0.sum(1).tolist(), m.sum(1).tolist()
            brows = rows[i:i + batch]
            per_row.setdefault(src, []).extend(
                [round(n, 6), len(r["assistant"]) + 1, int(k)] for n, k, r in zip(row_nats, row_toks, brows))
            if src == "temporal":
                am = answer_mask(yb, m)
                a_nats = ce.masked_fill(~am, 0.0).double().sum(1).tolist()
                a_toks = am.sum(1).tolist()
                for n_all, n_ans, k_ans, r in zip(row_nats, a_nats, a_toks, brows):
                    answer_rows.append([round(n_ans, 6), len(r["assistant"]), int(k_ans)])
                    t = by_task.setdefault(str(r.get("task", "?")), [0.0, 0, 0.0, 0])
                    t[0] += n_all
                    t[1] += len(r["assistant"]) + 1
                    t[2] += n_ans
                    t[3] += len(r["assistant"])
        key = "temporal" if src == "temporal" else f"val_{src}"
        rec[f"{key}_nats_per_char"] = nats / max(1, chars)
        rec[f"{key}_nats_per_token"] = nats / max(1, toks)
        if src != "temporal":
            pooled[0] += nats
            pooled[1] += chars
            pooled[2] += toks
    rec["val_nats_per_char_pooled"] = pooled[0] / max(1, pooled[1])
    rec["val_nats_per_token_pooled"] = pooled[0] / max(1, pooled[2])
    if by_task:
        rec["temporal_answer_nats_per_char"] = sum(t[2] for t in by_task.values()) / max(1, sum(t[3] for t in by_task.values()))
        rec["temporal_nats_per_char_by_task"] = {k: t[0] / max(1, t[1]) for k, t in sorted(by_task.items())}
        rec["temporal_answer_nats_per_char_by_task"] = {k: t[2] / max(1, t[3]) for k, t in sorted(by_task.items())}
    core = model.cns_full
    if core is not None:
        rec["gate_abs_mean"] = float(core.gate.detach().abs().mean())
        # only informative for gate="dynamic": a static gate gives every source the same mean |g|
        rec["gate_by_source"] = {s: float(np.mean(v)) for s, v in gate_by_src.items()}
        rec["state_autocorr"] = float(np.mean(autocorr)) if autocorr else None
        rec["mean_rate"] = float(np.mean(rates)) if rates else None
        rec["fraction_active"] = float(np.mean(active)) if active else None
        rec["write_rms_ratio"] = _weighted_mean(write_ratio)
        rec["write_token_var_frac"] = _weighted_mean(write_var)
    if row_losses is not None:
        row_losses["rows"] = per_row
        row_losses["keys_sha256"] = {
            s: hashlib.sha256("\n".join(ec.row_key(r["user"], r["assistant"]) for r in b["rows"]).encode()).hexdigest()
            for s, b in data["eval"].items()}
        if answer_rows:
            row_losses["temporal_answer"] = answer_rows
    if exact_match_items and data["temporal_heldout"]:
        hits: Dict[str, List[int]] = {}
        for r in data["temporal_heldout"][:exact_match_items]:
            ids, _ = tok.encode_turn(r["user"], None)
            if core is not None and hasattr(core, "reset_state"):
                core.reset_state()
            out = ec.greedy_decode(model, torch.tensor([ids]), max_new_tokens=12, **feats([r["user"]]))
            pred = tok.decode([t for t in out if t != text_utils.EOS])
            hits.setdefault(r["task"], []).append(int(tt.exact_match(pred, r["assistant"])))
        allh = [h for v in hits.values() for h in v]
        rec["temporal_exact_match"] = sum(allh) / max(1, len(allh))
        rec["temporal_exact_match_by_task"] = {k: sum(v) / len(v) for k, v in hits.items()}
    rec["eval_seconds"] = round(time.time() - t0, 1)
    return rec


@torch.no_grad()
def core_report(core) -> Dict[str, Any]:
    """Leak by superclass and edge drift of a lab core (the end-of-run anatomy summary)."""
    if core is None:
        return {}
    out: Dict[str, Any] = {}
    leak = torch.sigmoid(core.leak_logit).numpy()
    sc = (core.anatomy or {}).get("superclass")
    if sc:
        groups: Dict[str, List[float]] = {}
        for s, v in zip(sc, leak):
            groups.setdefault(s, []).append(float(v))
        out["leak_by_superclass"] = {s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                                     for s, v in sorted(groups.items())}
    out["leak_mean"] = float(leak.mean())
    if core.cfg.learned_edges:
        m = core.edge_multiplier().detach().numpy().astype(np.float64)
        w = core.edge_weights().detach().numpy()
        out["edge_multiplier"] = {"mean": float(m.mean()), "std": float(m.std()), "min": float(m.min()),
                                  "max": float(m.max()), "frac_changed_10pct": float((np.abs(m - 1) > 0.1).mean()),
                                  "sign_flips": int((np.sign(w) != np.sign(core.bio_weight.numpy())).sum())}
    if hasattr(core, "retain_logit"):
        out["retain_mean"] = float(torch.sigmoid(core.retain_logit).mean())
    return out


# ---------------------------------------------------------------------------
# final-eval ablations (protocol ``final_ablations: true``)
# ---------------------------------------------------------------------------
ABLATIONS = {"zero_edges": "val_nats_per_char_pooled_zero_edges", "gate0": "val_nats_per_char_pooled_gate0"}


@contextlib.contextmanager
def ablated_core(core, mode: str):
    """Temporarily ablate a lab core in place and restore it bit for bit afterwards.

    ``zero_edges``: ``bio_weight`` = 0, so every edge weight (bio x learned multiplier) is 0 and no
    signal crosses the graph (input and output types are disjoint): with ``state=reset`` the core's
    write is one constant vector (a persistent state relaxes towards it over the first tokens).
    ``gate0``: the gate (and a dynamic gate's projection) = 0: the graft is exactly absent.
    """
    if mode == "zero_edges":
        tensors = [core.bio_weight]
    elif mode == "gate0":
        tensors = [core.gate] + ([core.gate_proj.weight] if getattr(core, "gate_proj", None) is not None else [])
    else:
        raise ValueError(f"ablation {mode!r}; allowed {sorted(ABLATIONS)}")
    saved = [t.detach().clone() for t in tensors]
    try:
        with torch.no_grad():
            for t in tensors:
                t.zero_()
        core._op_cache = None  # the CSR operator caches bio_weight
        core._decode = None
        yield core
    finally:
        with torch.no_grad():
            for t, s in zip(tensors, saved):
                t.copy_(s)
        core._op_cache = None
        core._decode = None


@torch.no_grad()
def pooled_nats_per_char(model, data, feats, batch: int) -> float:
    """``val_nats_per_char_pooled`` exactly as :func:`evaluate` computes it (corpus sources only)."""
    model.eval()
    pooled = [0.0, 0]
    for src, block in data["eval"].items():
        if src == "temporal":
            continue
        nats = chars = 0
        x, y, rows = block["x"], block["y"], block["rows"]
        for i in range(0, x.shape[0], batch):
            _, ce, m = _eval_batch(model, feats, x, y, rows, i, batch)
            nats += float(ce[m].sum())
            chars += sum(len(r["assistant"]) + 1 for r in rows[i:i + batch])
        pooled[0] += nats
        pooled[1] += chars
    return pooled[0] / max(1, pooled[1])


def final_ablations(model, data, feats, batch: int, trained_pooled: float) -> Dict[str, Any]:
    """Pooled nats/char with the edges zeroed and with the gate closed, and the gains they imply."""
    core = model.cns_full
    if core is None:
        return {}
    out: Dict[str, Any] = {}
    for mode, key in ABLATIONS.items():
        with ablated_core(core, mode):
            out[key] = pooled_nats_per_char(model, data, feats, batch)
    # > 0: the part of the trained core's gain that needs signal to pass through the graph
    out["graph_dependent_gain"] = out[ABLATIONS["zero_edges"]] - float(trained_pooled)
    out["core_gain"] = out[ABLATIONS["gate0"]] - float(trained_pooled)  # the whole trained core's gain
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def store_final_rows(metrics: Dict[str, Any], rows_out: Optional[Dict[str, Any]]) -> None:
    """Per-row final-eval losses into ``metrics`` (``final_rows`` & co., see the module doc)."""
    if not rows_out:
        return
    metrics["final_rows"] = rows_out.get("rows", {})
    metrics["final_rows_sha256"] = rows_out.get("keys_sha256", {})
    if rows_out.get("temporal_answer"):
        metrics["final_rows_temporal_answer"] = rows_out["temporal_answer"]


def trainable_groups(model, core, P: Dict[str, Any]):
    scope = P.get("train_scope", "core")
    model.requires_grad_(False)
    groups = []
    if core is not None:
        core.requires_grad_(True)
        edge = core.edge_parameters()
        eid = {id(p) for p in edge}
        rest = [p for p in core.parameters() if id(p) not in eid]
        groups.append({"params": rest, "lr": float(P.get("lr_core", 1e-3)), "weight_decay": 0.0, "name": "core"})
        if edge:
            grp = {"params": edge, "lr": float(P.get("lr_edge", 3e-4)), "weight_decay": 0.0, "name": "edges"}
            if P.get("eps_edge") is not None:  # AdamW eps is per param group: only the edges get it
                grp["eps"] = float(P["eps_edge"])
            groups.append(grp)
    if scope in ("core+top", "all"):
        first = int(model.graft_layer) + 1 if scope == "core+top" else 0
        top = [p for layer in model.layers[first:] for p in layer.parameters()] + list(model.norm.parameters())
        if scope == "all":
            top += [p for n, p in model.named_parameters() if not n.startswith(("cns_full.", "layers.", "norm."))
                    and not n.startswith(("omni7.net.", "omni_core.collective", "omni_core.vision"))]
        for p in top:
            p.requires_grad_(True)
        groups.append({"params": top, "lr": float(P.get("lr_trunk", 3e-5)), "weight_decay": 0.01, "name": "trunk"})
    elif scope != "core":
        raise ValueError(f"train_scope {scope!r}")
    return [g for g in groups if g["params"]]


def run(h: Dict[str, Any], arm: str, seed: int, *, threads: int = 8, runs_root: Path = prereg.RUNS,
        smoke: bool = False, unlocked: bool = False) -> Dict[str, Any]:
    P = dict(h["protocol"])
    if smoke:
        P.update({"steps": 3, "eval_steps": [3], "eval_per_source": 4,
                  "temporal": {**(P.get("temporal") or {}), "eval_rows": 8, "train_rows": 16, "exact_match_items": 4}})
    cfg = prereg.arm_config(h, arm)
    rdir = Path(runs_root) / (prereg.run_key(P, cfg, seed) + ("_smoke" if smoke else ""))
    mpath = rdir / "metrics.json"
    if mpath.exists() and json.loads(mpath.read_text(encoding="utf-8")).get("status") == "done":
        log(f"{h['id']} {arm} seed {seed}: already done ({rdir.name})")
        return json.loads(mpath.read_text(encoding="utf-8"))
    rdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    t_start = time.time()
    steps, batch = int(P["steps"]), int(P.get("batch", 8))
    eval_steps = sorted({int(s) for s in P.get("eval_steps", [steps])} | {steps})
    metrics: Dict[str, Any] = {"schema": METRICS_SCHEMA, "hypothesis": h["id"], "arm": arm, "seed": int(seed),
                               "run_key": rdir.name, "config": cfg.to_dict(), "protocol": P, "status": "running",
                               "unlocked": bool(unlocked), "lock": h.get("_lock"), "evals": [], "train_log": []}
    write_json(mpath, metrics)

    base = ROOT / P.get("base_checkpoint", "checkpoints/supermix_expanse_v3.pt")
    model, tok, payload = ec.load_expanse(base)
    payload["state_dict"] = None
    graph, _ = cl.load_or_build_lab_graph(cfg, seed, cache_dir=prereg.OUT / "graphs")
    core = cl.fresh_lab_core(int(model.config.hidden_size), graph, cfg, seed)
    ec.attach_lab_core(model, core)
    groups = trainable_groups(model, core, P)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    metrics["graph"] = graph["receipt"] if graph is not None else None
    metrics["trainable_params"] = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}
    log(f"{h['id']} arm {arm} seed {seed}: {cfg.to_dict()} | graph "
        f"{'none' if graph is None else str(graph['n']) + ' types / ' + format(len(graph['post']), ',') + ' edges'}"
        f" | trainable {n_train:,}")

    data = build_data(model, tok, P, seed)
    metrics["data"] = data["counts"]
    feats = Features(model, payload)
    tpb = int((P.get("temporal") or {}).get("per_batch", 0))
    plan = batch_plan(len(data["m_tr"]), len(data["m_tt"]), steps, batch, tpb, seed)
    em_items = int((P.get("temporal") or {}).get("exact_match_items", 0))
    log(f"data: {data['counts']}")

    untrainable = not groups
    metrics["untrainable"] = untrainable
    if untrainable:
        rows_out: Dict[str, Any] = {}
        rec = evaluate(model, tok, data, feats, batch, 0, em_items, row_losses=rows_out)
        for s in eval_steps:
            metrics["evals"].append({**rec, "step": s})
        metrics["steps_done"] = 0
        store_final_rows(metrics, rows_out)
    else:
        opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups], betas=(0.9, 0.95))
        metrics["optimizer_eps"] = {g["name"]: float(pg["eps"]) for g, pg in zip(groups, opt.param_groups)}
        base_lrs = [g["lr"] for g in groups]
        warm = max(1, int(float(P.get("warmup_frac", 0.05)) * steps))
        floor = float(P.get("min_lr_frac", 0.05))

        def lr_scale(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(1, steps - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

        params = [p for g in groups for p in g["params"]]
        step_times: List[float] = []
        tokens = 0
        if P.get("eval_at_start", False):
            metrics["evals"].append(evaluate(model, tok, data, feats, batch, 0))
        model.eval()  # frozen trunk runs as at inference; gradients still flow into the core
        for step in range(steps):
            pl = plan[step]
            xs, ys, users = [data["x_tr"][pl["corpus"]]], [data["y_tr"][pl["corpus"]]], [data["m_tr"][int(i)]["user"] for i in pl["corpus"]]
            if pl["temporal"] is not None:
                xs.append(data["x_tt"][pl["temporal"]])
                ys.append(data["y_tt"][pl["temporal"]])
                users += [data["m_tt"][int(i)]["user"] for i in pl["temporal"]]
            xb, yb = torch.cat(xs), torch.cat(ys)
            # trim to the longest row: fewer PAD columns through the core and the trunk
            L = int((xb != text_utils.PAD).sum(1).max())
            xb, yb = xb[:, :L], yb[:, :L]
            for grp, lr in zip(opt.param_groups, base_lrs):
                grp["lr"] = lr * lr_scale(step)
            t0 = time.perf_counter()
            with torch.enable_grad():
                lg = model(xb, return_mtp=False, **feats(users)).logits
                loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), yb[:, 1:].reshape(-1), ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, float(P.get("grad_clip", 1.0))))
            opt.step()
            step_times.append(time.perf_counter() - t0)
            tokens += int((xb != text_utils.PAD).sum())
            if not math.isfinite(float(loss)):
                metrics["status"] = "failed"
                metrics["error"] = f"non-finite loss at step {step + 1}"
                write_json(mpath, metrics)
                raise FloatingPointError(metrics["error"])
            if (step + 1) % 10 == 0 or step == 0:
                recent = step_times[-10:]
                g_abs = float(core.gate.detach().abs().mean()) if core is not None else 0.0
                metrics["train_log"].append({"step": step + 1, "loss": float(loss), "grad_norm": gnorm,
                                             "gate_abs_mean": g_abs, "s_per_step": sum(recent) / len(recent)})
                log(f"[step {step + 1}/{steps}] loss {float(loss):.4f} gnorm {gnorm:.3f} |gate| {g_abs:.4f} "
                    f"{sum(recent) / len(recent):.2f}s/step eta {(steps - step - 1) * sum(recent) / len(recent) / 60:.1f}m")
            if step + 1 in eval_steps:
                final = step + 1 == steps
                rows_out = {} if final else None
                rec = evaluate(model, tok, data, feats, batch, step + 1, em_items if final else 0, row_losses=rows_out)
                if final and P.get("final_ablations"):
                    t_ab = time.time()
                    rec.update(final_ablations(model, data, feats, batch, rec["val_nats_per_char_pooled"]))
                    rec["ablation_seconds"] = round(time.time() - t_ab, 1)
                metrics["evals"].append(rec)
                metrics["steps_done"] = step + 1
                if final:
                    store_final_rows(metrics, rows_out)
                log(f"[eval {step + 1}] pooled {rec['val_nats_per_char_pooled']:.4f} nats/char"
                    + (f", temporal {rec['temporal_nats_per_char']:.4f}" if "temporal_nats_per_char" in rec else "")
                    + (f", temporal EM {rec['temporal_exact_match']:.3f}" if "temporal_exact_match" in rec else "")
                    + (f", graph-dependent gain {rec['graph_dependent_gain']:+.5f}" if "graph_dependent_gain" in rec else ""))
                write_json(mpath, metrics)
        metrics["steps_done"] = steps
        metrics["throughput"] = {"s_per_step_mean": sum(step_times) / len(step_times),
                                 "s_per_step_median": float(np.median(step_times)),
                                 "train_tokens_per_s": tokens / max(1e-9, sum(step_times))}
    metrics["core_report"] = core_report(core)
    metrics["peak_rss_mb"] = peak_rss_mb()
    metrics["wall_minutes"] = round((time.time() - t_start) / 60, 2)
    if core is not None and not smoke:
        cl.save_core_file(rdir / "core.pt", core, {"hypothesis": h["id"], "arm": arm, "seed": seed},
                          keep_constant=core.nnz <= 1_500_000)
    metrics["status"] = "done"
    write_json(mpath, metrics)
    log(f"done {h['id']} {arm} seed {seed} in {metrics['wall_minutes']} min -> {mpath}")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--hyp_dir", default=str(prereg.HYP_DIR))
    ap.add_argument("--runs", default=str(prereg.RUNS))
    ap.add_argument("--smoke", action="store_true", help="3 steps, tiny eval (implies nothing about the hypothesis)")
    ap.add_argument("--unlocked", action="store_true", help="allow an unlocked hypothesis (smoke / development only)")
    args = ap.parse_args(argv)
    if args.unlocked:
        h = json.loads(prereg.hyp_path(args.hypothesis, Path(args.hyp_dir)).read_text(encoding="utf-8"))
        prereg.validate(h)
        if not args.smoke:
            raise SystemExit("--unlocked is for --smoke runs only: lock the hypothesis first")
    else:
        h = prereg.load_locked(args.hypothesis, Path(args.hyp_dir))
    run(h, args.arm, args.seed, threads=args.threads, runs_root=Path(args.runs), smoke=args.smoke,
        unlocked=args.unlocked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
