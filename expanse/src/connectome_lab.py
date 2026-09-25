"""Expanse Connectome Laboratory: the male-CNS core as a controlled experiment.

``connectome_full.FullConnectomeCore`` runs the real 11,751-type male-CNS graph
with fixed, measured weights. This module generalises it so that *what the
wiring contributes* can be measured instead of assumed (spec:
``CNS_LAB_DESIGN.md``; method after QuixiAI/FlyGPT). Every experimental axis is
one field of :class:`LabConfig`:

    topology       real | shuffled | random | none
    sign_mode      dale | unsigned
    strength_init  bio | random
    edge_param     fixed | residual | dale_softplus
    norm           frac | sqrt_indeg | raw
    activation     relu | tanh
    state          reset | persist | window:K | learned_decay
    microsteps     recurrent updates per token
    gate           static | dynamic
    subgraph       None | N cell types
    backend        csr_slab | torch_sparse | dense

``LabConfig()`` (all defaults) is exactly the v3 core: :func:`lab_from_full`
copies a trained ``FullConnectomeCore`` into a lab core whose output is
bitwise the same, and the learned-edge forms start at exactly the same
function (``residual``: ``W_bio * (1 + alpha)``, alpha = 0; ``dale_softplus``:
``W_bio * softplus(theta) / softplus(theta0)``, theta = theta0, so each edge's
multiplier is exactly 1 at birth and, being positive, can never flip Dale's
sign).

Graph construction (:func:`lab_graph`, numpy) runs in this order on a
``build_full_graph`` dict: optional subgraph -> per-edge magnitude from the
synapse counts of the *real* graph (``frac``: share of the post type's input,
the v3 weights; ``sqrt_indeg``: count / sqrt(in-degree), FlyGPT's
normalisation; ``raw``: the count) -> topology (``shuffled``: the
degree-preserving null of ``connectome_full.rewired_graph``, magnitudes carried
with their target so every type keeps its in-degree, out-degree and multiset of
input magnitudes; ``random``: uniform random pairs, same edge count, the same
magnitudes dealt at random) -> ``strength_init="random"`` (magnitudes
permuted across edges) -> sign (Dale: the presynaptic type's; unsigned: +1)
-> scale. Row-stochastic magnitudes (``frac`` on the real or shuffled graph:
every row of ``|W|`` sums to 1) are multiplied by ``radius`` exactly (bitwise the
v3 values; Perron root <= ``radius``, and = 0.9 measured on the real graph);
every other variant is rescaled so a measured power iteration gives
``radius``. Arms therefore differ in *structure*, not in gain. Every per-type ``g_in`` is a learned input
normalisation in all variants ("learned" normalisation of the design notes).

Dynamics per token column (``a = sigmoid(leak)``, ``act`` relu or tanh)::

    r <- (1 - a) r + a act(g_in * W (g_out * r) + U + b)      (microsteps times)

``state="reset"`` starts every token at r = 0 (v3; all tokens run as one batch
of columns). The persistent modes run a causal scan over positions and carry
``r`` from one token to the next: ``persist`` as is, ``window:K`` reset at
every absolute position divisible by K, ``learned_decay`` multiplied by a
learned per-type retention ``sigmoid(retain_logit)`` (0.5 at birth) between
tokens. PAD positions (``token_mask`` False) carry the state unchanged and are
not written. A cached decode step (``past_length > 0``) continues from the
state the previous call stored, so KV-cached decoding equals the full forward.

``gate="dynamic"`` adds an input-dependent term to the per-channel gate:
``g(h) = gate + tanh(W_g RMSNorm(h))`` with ``W_g`` zero at birth.

Per-edge learning (``residual`` / ``dale_softplus``) needs ``dL/dW_e =
sum_c grad_y[post_e, c] x[pre_e, c]`` for every spmm. The ``csr_slab``
backend defers these: each spmm's backward only records ``(grad_y, x)``, and
one sampled dense-dense product (``torch.sparse.sampled_addmm``) over all the
recorded columns runs once, in an anchor node every recurrent chain starts
from, so autograd runs it after every spmm's backward. For the persistent scan
this is one call over T x (microsteps - 1) x B columns instead of hundreds of
small ones.

Anatomy: the core keeps ``edge_dst``/``col`` (= ``edge_src``), ``synapse_count``
and ``bio_weight`` as buffers (CSR order of W) and node metadata (type names,
superclass, NT, neuron counts, input/output sets, subgraph members) in
:meth:`ConnectomeLabCore.spec`, which Expanse checkpoints save.
:func:`export_anatomy` adds the learned weights and ``learned / bio``
multipliers.

CLI::

    python src/connectome_lab.py export <expanse ckpt | lab core file> <out.npz>
    python src/connectome_lab.py graph --topology shuffled --subgraph 3000 --seed 1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import connectome_full as cf

LAB_SCHEMA = "supermix-expanse-cns-lab-v1"
CORE_FILE_SCHEMA = "supermix-expanse-cns-lab-core-v1"
LAB_GRAPH_SCHEMA = "supermix-expanse-cns-lab-graph-v1"
THETA0 = math.log(math.e - 1.0)  # softplus(THETA0) == 1

TOPOLOGIES = ("real", "shuffled", "random", "none")
SIGN_MODES = ("dale", "unsigned")
STRENGTH_INITS = ("bio", "random")
EDGE_PARAMS = ("fixed", "residual", "dale_softplus")
NORMS = ("frac", "sqrt_indeg", "raw")
ACTIVATIONS = ("relu", "tanh")
GATES = ("static", "dynamic")
BACKENDS = ("csr_slab", "torch_sparse", "dense")
#: Columns per sampled_addmm call when edge gradients are flushed.
SDDMM_CHUNK = 1024


def npz_path() -> Path:
    return Path(os.environ.get("MALECNS_TYPES_NPZ") or cf.DEFAULT_NPZ)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabConfig:
    """One point in the experiment space. Defaults reproduce the v3 core exactly."""

    topology: str = "real"
    sign_mode: str = "dale"
    strength_init: str = "bio"
    edge_param: str = "fixed"
    norm: str = "frac"
    activation: str = "relu"
    state: str = "reset"
    microsteps: int = 4
    gate: str = "static"
    subgraph: Optional[int] = None
    radius: float = 0.9
    graph_seed: Optional[int] = None  # shuffled / random / strength draws; None = the run seed
    backend: str = "csr_slab"

    def __post_init__(self):
        for name, allowed in (("topology", TOPOLOGIES), ("sign_mode", SIGN_MODES), ("strength_init", STRENGTH_INITS),
                              ("edge_param", EDGE_PARAMS), ("norm", NORMS), ("activation", ACTIVATIONS),
                              ("gate", GATES), ("backend", BACKENDS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"LabConfig.{name}={getattr(self, name)!r}; allowed {allowed}")
        if self.state not in ("reset", "persist", "learned_decay"):
            if not (self.state.startswith("window:") and self.state[7:].isdigit() and int(self.state[7:]) >= 1):
                raise ValueError(f"LabConfig.state={self.state!r}; allowed reset | persist | window:K | learned_decay")
        if int(self.microsteps) < 1:
            raise ValueError("microsteps must be >= 1")
        if self.subgraph is not None and int(self.subgraph) < 8:
            raise ValueError("subgraph must keep at least 8 types")
        if not (0.0 < float(self.radius) < 1.0):
            raise ValueError("radius must be in (0, 1) for contractive dynamics")

    @property
    def window(self) -> int:
        return int(self.state[7:]) if self.state.startswith("window:") else 0

    @property
    def persistent(self) -> bool:
        return self.state != "reset"

    @property
    def learned_edges(self) -> bool:
        return self.edge_param != "fixed"

    def graph_seed_for(self, run_seed: int) -> int:
        return int(self.graph_seed) if self.graph_seed is not None else int(run_seed)

    def graph_fields(self) -> Dict[str, Any]:
        """The fields that determine W (the rest shape the dynamics or the parameters)."""
        return {k: getattr(self, k) for k in ("topology", "sign_mode", "strength_init", "norm", "subgraph", "radius")}

    def graph_is_seeded(self) -> bool:
        return self.topology in ("shuffled", "random") or self.strength_init == "random"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LabConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown LabConfig fields {sorted(unknown)}")
        return cls(**d)

    def replace(self, **kw) -> "LabConfig":
        return dataclasses.replace(self, **kw)


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------
def _meta(base: dict, key: str, n: int, fill: Any) -> np.ndarray:
    if key in base and base[key] is not None:
        return np.asarray(base[key])
    return np.array([fill] * n, dtype=object)


def select_subgraph(base: dict, n_keep: int) -> np.ndarray:
    """``n_keep`` type indices (sorted): a synapse-mass core plus the inputs/outputs that talk to it.

    Deterministic. The input and output sets keep their share of the full graph
    (1,277 / 713 of 11,751): first the ``other`` types with the most synapses
    (in + out) form the core ``C``; then the input types sending the most
    synapses into ``C`` and the output types receiving the most from ``C``.
    """

    n = int(base["n"])
    n_keep = int(n_keep)
    if n_keep >= n:
        return np.arange(n, dtype=np.int64)
    post = np.asarray(base["post"], dtype=np.int64)
    pre = np.asarray(base["pre"], dtype=np.int64)
    w = np.asarray(base["weight"], dtype=np.float64) if "weight" in base else np.abs(np.asarray(base["value"], np.float64))
    in_idx = np.asarray(base["in_idx"], dtype=np.int64)
    out_idx = np.asarray(base["out_idx"], dtype=np.int64)
    n_in_k = max(1, int(round(n_keep * len(in_idx) / n)))
    n_out_k = max(1, int(round(n_keep * len(out_idx) / n)))
    n_mid = n_keep - n_in_k - n_out_k
    if n_mid < 1:
        raise ValueError(f"subgraph {n_keep} too small for its input/output shares")
    role = cf.source_roles(base)
    mass = np.bincount(post, weights=w, minlength=n) + np.bincount(pre, weights=w, minlength=n)
    other = np.flatnonzero(role == cf.ROLE_OTHER)
    core = other[np.argsort(-mass[other], kind="stable")[:n_mid]]
    in_core = np.zeros(n, dtype=bool)
    in_core[core] = True
    to_core = np.bincount(pre[in_core[post]], weights=w[in_core[post]], minlength=n)
    from_core = np.bincount(post[in_core[pre]], weights=w[in_core[pre]], minlength=n)
    ins = in_idx[np.argsort(-to_core[in_idx], kind="stable")[:n_in_k]]
    outs = out_idx[np.argsort(-from_core[out_idx], kind="stable")[:n_out_k]]
    return np.unique(np.concatenate([core, ins, outs])).astype(np.int64)


def _random_pairs(n: int, e: int, rng: np.random.Generator) -> np.ndarray:
    """``e`` distinct keys ``post * n + pre`` drawn uniformly from all n^2 ordered pairs."""
    if e > n * n:
        raise ValueError("more edges than ordered pairs")
    keys = np.unique(rng.integers(0, n * n, size=int(e * 1.05) + 16, dtype=np.int64))
    while len(keys) < e:
        keys = np.unique(np.concatenate([keys, rng.integers(0, n * n, size=e - len(keys) + 16, dtype=np.int64)]))
    return np.sort(rng.choice(keys, size=e, replace=False))


def arrays_sha256(post: np.ndarray, pre: np.ndarray, value: np.ndarray) -> str:
    h = hashlib.sha256()
    for a, dt in ((post, np.int32), (pre, np.int32), (value, np.float32)):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=dt)).tobytes())
    return h.hexdigest()


def lab_graph(base: dict, cfg: LabConfig, seed: int = 0) -> Optional[dict]:
    """The graph a lab arm runs, from a ``build_full_graph`` dict. ``None`` for ``topology="none"``.

    Returns the ``build_full_graph`` keys (``n, post, pre, value, sign,
    in_idx, out_idx, type_names, superclass, nt, weight, n_neurons, receipt``)
    plus ``kept`` (original type index of each node) and ``lab_config``.
    ``weight`` is each edge's synapse count, moved with its magnitude by the
    topology / strength transforms.
    """

    if cfg.topology == "none":
        return None
    t0 = time.time()
    gseed = cfg.graph_seed_for(seed)
    n0 = int(base["n"])
    post = np.asarray(base["post"], dtype=np.int64)
    pre = np.asarray(base["pre"], dtype=np.int64)
    w = (np.asarray(base["weight"], dtype=np.float64) if base.get("weight") is not None
         else np.abs(np.asarray(base["value"], dtype=np.float64)))
    type_sign = np.asarray(base["sign"], dtype=np.int8)
    in_idx = np.asarray(base["in_idx"], dtype=np.int64)
    out_idx = np.asarray(base["out_idx"], dtype=np.int64)
    names = _meta(base, "type_names", n0, "")
    superclass = _meta(base, "superclass", n0, "")
    nt = _meta(base, "nt", n0, "")
    n_neurons = np.asarray(base["n_neurons"], dtype=np.int64) if base.get("n_neurons") is not None else np.zeros(n0, np.int64)

    kept = np.arange(n0, dtype=np.int64)
    sub_info: Optional[Dict[str, Any]] = None
    if cfg.subgraph is not None and int(cfg.subgraph) < n0:
        kept = select_subgraph(base, int(cfg.subgraph))
        new_id = np.full(n0, -1, dtype=np.int64)
        new_id[kept] = np.arange(len(kept))
        keep_e = (new_id[post] >= 0) & (new_id[pre] >= 0)
        mass_all = float(w.sum())
        post, pre, w = new_id[post[keep_e]], new_id[pre[keep_e]], w[keep_e]
        in_idx = new_id[in_idx][new_id[in_idx] >= 0]
        out_idx = new_id[out_idx][new_id[out_idx] >= 0]
        type_sign, names, superclass, nt, n_neurons = (a[kept] for a in (type_sign, names, superclass, nt, n_neurons))
        sub_info = {"n_types": int(len(kept)), "n_edges": int(len(post)),
                    "synapse_mass_kept": round(float(w.sum()) / max(mass_all, 1e-12), 6),
                    "selection": "top other types by synapse mass + inputs into / outputs from them"}
    n = int(len(kept))
    e = int(len(post))

    # magnitudes from the real (sub)graph's synapse counts
    if cfg.norm == "frac":
        total_in = np.bincount(post, weights=w, minlength=n)
        mag = w / total_in[post]
    elif cfg.norm == "sqrt_indeg":
        indeg = np.bincount(post, minlength=n).astype(np.float64)
        mag = w / np.sqrt(indeg[post])
    else:
        mag = w.copy()

    topo_info: Dict[str, Any] = {}
    real_keys = np.sort(post * n + pre)
    if cfg.topology == "shuffled":
        role = cf.source_roles({"n": n, "in_idx": in_idx, "out_idx": out_idx})
        stratum = ((type_sign > 0).astype(np.int64) * 3 + role)[pre]
        new_post, carried, topo_info = cf._vectorised_stratified_swaps(post, pre, np.arange(e, dtype=np.int64),
                                                                        stratum, n, rounds=20, seed=gseed)
        post, mag, w = new_post, mag[carried], w[carried]
    elif cfg.topology == "random":
        rng = np.random.default_rng([gseed, 1])
        keys = _random_pairs(n, e, rng)
        post, pre = keys // n, keys % n
        perm = rng.permutation(e)
        mag, w = mag[perm], w[perm]
        topo_info = {"sampler": "uniform distinct ordered pairs; magnitudes dealt by a random permutation"}
    if cfg.strength_init == "random":
        perm = np.random.default_rng([gseed, 2]).permutation(e)
        mag, w = mag[perm], w[perm]

    order = np.argsort(post * n + pre, kind="stable")
    post, pre, mag, w = post[order], pre[order], mag[order], w[order]
    sgn = type_sign[pre].astype(np.float64) if cfg.sign_mode == "dale" else np.ones(e, dtype=np.float64)
    row_stochastic = cfg.norm == "frac" and cfg.topology in ("real", "shuffled") and cfg.strength_init == "bio"
    if row_stochastic:
        scale = float(cfg.radius)  # rows of |W| sum to radius (Perron root <= radius): bitwise the v3 values
        rho_raw = None
    else:
        rho_raw = cf.abs_spectral_radius(post, pre, mag, n)["estimate"]
        scale = float(cfg.radius) / max(rho_raw, 1e-12)
    value = (mag * sgn * scale).astype(np.float32)
    rho = cf.abs_spectral_radius(post, pre, value, n)

    keys = post * n + pre
    pos = np.minimum(np.searchsorted(real_keys, keys), max(0, len(real_keys) - 1))
    overlap = float((real_keys[pos] == keys).mean()) if e else 0.0
    receipt = {
        "schema": LAB_GRAPH_SCHEMA,
        "config": cfg.to_dict(),
        "graph_seed": gseed if cfg.graph_is_seeded() else None,
        "base_npz_sha256": (base.get("receipt") or {}).get("npz_sha256"),
        "n_types": n, "n_edges": e, "n_in": int(len(in_idx)), "n_out": int(len(out_idx)),
        "subgraph": sub_info,
        "topology_info": cf._jsonable(topo_info),
        "edge_overlap_with_real": round(overlap, 6),
        "scale": scale, "abs_spectral_radius_before_scale": rho_raw,
        "abs_spectral_radius": float(rho["estimate"]),
        "edge_sign_counts": {"positive": int((value > 0).sum()), "negative": int((value < 0).sum())},
        "arrays_sha256": arrays_sha256(post, pre, value),
        "seconds": round(time.time() - t0, 2),
    }
    return {
        "n": n, "post": post.astype(np.int32), "pre": pre.astype(np.int32), "value": value,
        "sign": type_sign.astype(np.int8), "in_idx": in_idx.astype(np.int64), "out_idx": out_idx.astype(np.int64),
        "type_names": names, "superclass": superclass, "nt": nt, "n_neurons": n_neurons.astype(np.int64),
        "weight": w, "kept": kept if cfg.subgraph is not None else None, "receipt": receipt,
        "lab_config": cfg.to_dict(),
    }


def load_or_build_lab_graph(cfg: LabConfig, seed: int, cache_dir: Optional[os.PathLike] = None,
                            base: Optional[dict] = None) -> Tuple[Optional[dict], Optional[dict]]:
    """(lab graph, base graph or None) with an npz cache keyed by the npz hash, graph fields and seed."""
    if cfg.topology == "none":
        return None, base
    src = npz_path()
    key_obj = {"npz": cf.sha256_file(src) if base is None else (base.get("receipt") or {}).get("npz_sha256"),
               "fields": cfg.graph_fields(), "seed": cfg.graph_seed_for(seed) if cfg.graph_is_seeded() else None,
               "schema": LAB_GRAPH_SCHEMA}
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"lab_graph_{key}.npz" if cache_dir else None
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        g = {k: d[k] for k in ("post", "pre", "value", "sign", "in_idx", "out_idx", "weight", "n_neurons")}
        g["n"] = int(d["n"])
        for k in ("type_names", "superclass", "nt"):
            g[k] = d[k].astype(object)
        g["kept"] = d["kept"] if d["kept"].size else None
        g["receipt"] = json.loads(str(d["receipt"]))
        g["lab_config"] = json.loads(str(d["lab_config"]))
        # the cached arrays must still be the graph this config describes
        if g["receipt"]["arrays_sha256"] != arrays_sha256(g["post"], g["pre"], g["value"]):
            raise RuntimeError(f"{path} is corrupt (array hash mismatch)")
        return g, base
    if base is None:
        base = cf.build_full_graph(src)
    g = lab_graph(base, cfg, seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, n=np.int64(g["n"]), post=g["post"], pre=g["pre"], value=g["value"], sign=g["sign"],
                 in_idx=g["in_idx"], out_idx=g["out_idx"], weight=g["weight"], n_neurons=g["n_neurons"],
                 type_names=np.asarray(g["type_names"], dtype=str), superclass=np.asarray(g["superclass"], dtype=str),
                 nt=np.asarray(g["nt"], dtype=str),
                 kept=np.asarray(g["kept"] if g["kept"] is not None else np.zeros(0), dtype=np.int64),
                 receipt=np.asarray(json.dumps(cf._jsonable(g["receipt"]))),
                 lab_config=np.asarray(json.dumps(g["lab_config"])))
        os.replace(tmp, path)
    return g, base


# ---------------------------------------------------------------------------
# sparse operators with deferred edge gradients
# ---------------------------------------------------------------------------
def sddmm(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Values of ``a @ b^T`` at the non-zeros of CSR ``w`` (in ``w``'s order). ``a``, ``b``: (n, M)."""
    return torch.sparse.sampled_addmm(w, a.contiguous(), b.t().contiguous(), beta=0.0).values()


class EdgeGradCollector:
    """Records ``(grad_y, x)`` of every spmm that used ``w``; :meth:`grad` sums them in one pass."""

    def __init__(self, w: torch.Tensor, chunk: int = SDDMM_CHUNK):
        self.w = w
        self.chunk = int(chunk)
        self.pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def grad(self) -> torch.Tensor:
        pairs, self.pairs = self.pairs, []
        out: Optional[torch.Tensor] = None
        buf_g: List[torch.Tensor] = []
        buf_x: List[torch.Tensor] = []
        cols = 0

        def flush():
            nonlocal out, buf_g, buf_x, cols
            if not buf_g:
                return
            part = sddmm(self.w, torch.cat(buf_g, 1), torch.cat(buf_x, 1))
            out = part if out is None else out + part
            buf_g, buf_x, cols = [], [], 0

        for gy, x in pairs:
            buf_g.append(gy)
            buf_x.append(x)
            cols += gy.shape[1]
            if cols >= self.chunk:
                flush()
        flush()
        if out is None:
            out = torch.zeros(self.w.values().shape, dtype=self.w.dtype, device=self.w.device)
        return out


class _EdgeSpMM(torch.autograd.Function):
    """``y = W x``; backward returns ``W^T grad_y`` and hands ``(grad_y, x)`` to the collector."""

    @staticmethod
    def forward(ctx, x, w, w_t, collector):  # type: ignore[override]
        ctx.w_t = w_t
        ctx.collector = collector
        ctx.save_for_backward(x)
        return cf.spmm(w, x)

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        ctx.collector.pairs.append((grad_y.detach(), x.detach()))
        return cf.spmm(ctx.w_t, grad_y), None, None, None


class _EdgeGradAnchor(torch.autograd.Function):
    """Returns ``zeros``; every recurrent chain adds it to its first state, so its backward
    runs after every spmm's backward and turns the collected pairs into ``dL/dvalues``."""

    @staticmethod
    def forward(ctx, values, zeros, collector):  # type: ignore[override]
        ctx.collector = collector
        return zeros.clone()

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return ctx.collector.grad(), None, None


# ---------------------------------------------------------------------------
# the core
# ---------------------------------------------------------------------------
class ConnectomeLabCore(nn.Module):
    """A configurable male-CNS core (see module doc). ``graph`` is a :func:`lab_graph` /
    ``build_full_graph`` dict, or a shape-only ``{"n", "n_in", "n_out", "nnz"}`` when the
    buffers come from ``load_state_dict``."""

    wants_past_length = True
    CONST_BUFFERS = ("crow", "col", "crow_t", "col_t", "perm_t", "edge_dst", "bio_weight", "synapse_count",
                     "in_idx", "out_idx", "type_sign")

    def __init__(self, hidden_size: int, graph: dict, cfg: Optional[LabConfig] = None):
        super().__init__()
        cfg = cfg if isinstance(cfg, LabConfig) else LabConfig.from_dict(cfg)
        if cfg.topology == "none":
            raise ValueError("topology 'none' has no core: attach_lab_core(model, None)")
        self.cfg = cfg
        shapes = cf.graph_shapes(graph)
        self.hidden_size = h = int(hidden_size)
        self.n = n = shapes["n"]
        self.n_in = shapes["n_in"]
        self.n_out = shapes["n_out"]
        self.nnz = nnz = shapes["nnz"]
        self.steps = int(cfg.microsteps)
        for name, dt, size in (("crow", torch.int32, n + 1), ("col", torch.int32, nnz), ("crow_t", torch.int32, n + 1),
                               ("col_t", torch.int32, nnz), ("perm_t", torch.int64, nnz), ("edge_dst", torch.int32, nnz),
                               ("bio_weight", torch.float32, nnz), ("synapse_count", torch.int32, nnz)):
            self.register_buffer(name, torch.zeros(size, dtype=dt))
        self.register_buffer("in_idx", torch.zeros(self.n_in, dtype=torch.int64))
        self.register_buffer("out_idx", torch.zeros(self.n_out, dtype=torch.int64))
        self.register_buffer("type_sign", torch.ones(n, dtype=torch.int8))

        self.in_norm = nn.RMSNorm(h)
        self.in_proj = nn.Linear(h, self.n_in, bias=False)
        self.out_proj = nn.Linear(self.n_out, h, bias=False)
        self.log_gain_in = nn.Parameter(torch.zeros(n))
        self.log_gain_out = nn.Parameter(torch.zeros(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.leak_logit = nn.Parameter(torch.zeros(n))
        self.gate = nn.Parameter(torch.zeros(h))
        if cfg.edge_param == "residual":
            self.edge_alpha = nn.Parameter(torch.zeros(nnz))
        elif cfg.edge_param == "dale_softplus":
            self.edge_theta = nn.Parameter(torch.full((nnz,), THETA0))
        if cfg.state == "learned_decay":
            self.retain_logit = nn.Parameter(torch.zeros(n))  # retention 0.5 at birth
        if cfg.gate == "dynamic":
            self.gate_norm = nn.RMSNorm(h)
            self.gate_proj = nn.Linear(h, h, bias=False)
            nn.init.zeros_(self.gate_proj.weight)
        self.anatomy: Dict[str, Any] = {}
        self.graph_receipt: Dict[str, Any] = {}
        self._op_cache: Optional[Tuple[Any, ...]] = None
        self._decode: Optional[Dict[str, Any]] = None
        if "post" in graph:
            self.set_graph(graph)

    # -- graph ---------------------------------------------------------------
    @torch.no_grad()
    def set_graph(self, graph: dict) -> None:
        shapes = cf.graph_shapes(graph)
        if (shapes["n"], shapes["n_in"], shapes["n_out"], shapes["nnz"]) != (self.n, self.n_in, self.n_out, self.nnz):
            raise ValueError(f"graph {shapes} does not match the core ({self.n}, {self.n_in}, {self.n_out}, {self.nnz})")
        n = self.n
        post = np.asarray(graph["post"], dtype=np.int64)
        pre = np.asarray(graph["pre"], dtype=np.int64)
        value = np.asarray(graph["value"], dtype=np.float32)
        weight = graph.get("weight")
        ord_w = np.argsort(post * n + pre, kind="stable")
        ord_t = np.argsort(pre * n + post, kind="stable")
        pos_w = np.empty(len(ord_w), dtype=np.int64)
        pos_w[ord_w] = np.arange(len(ord_w))
        crow = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(post, minlength=n), out=crow[1:])
        crow_t = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=crow_t[1:])
        dev = self.bio_weight.device
        self.crow = torch.from_numpy(crow.astype(np.int32)).to(dev)
        self.col = torch.from_numpy(pre[ord_w].astype(np.int32)).to(dev)
        self.edge_dst = torch.from_numpy(post[ord_w].astype(np.int32)).to(dev)
        self.bio_weight = torch.from_numpy(value[ord_w].copy()).to(dev)
        self.crow_t = torch.from_numpy(crow_t.astype(np.int32)).to(dev)
        self.col_t = torch.from_numpy(post[ord_t].astype(np.int32)).to(dev)
        self.perm_t = torch.from_numpy(pos_w[ord_t]).to(dev)
        sc = np.zeros(len(ord_w), dtype=np.int32) if weight is None else np.rint(np.asarray(weight, np.float64)[ord_w]).astype(np.int32)
        self.synapse_count = torch.from_numpy(sc).to(dev)
        self.in_idx = torch.as_tensor(np.asarray(graph["in_idx"], dtype=np.int64)).to(dev)
        self.out_idx = torch.as_tensor(np.asarray(graph["out_idx"], dtype=np.int64)).to(dev)
        self.type_sign = torch.as_tensor(np.asarray(graph["sign"], dtype=np.int8)).to(dev)

        def as_list(key, cast):
            v = graph.get(key)
            return None if v is None else [cast(x) for x in np.asarray(v).tolist()]

        self.anatomy = {
            "node_id": as_list("type_names", str), "superclass": as_list("superclass", str), "nt": as_list("nt", str),
            "n_neurons": as_list("n_neurons", int), "type_sign": [int(x) for x in np.asarray(graph["sign"]).tolist()],
            "input_nodes": [int(x) for x in np.asarray(graph["in_idx"]).tolist()],
            "output_nodes": [int(x) for x in np.asarray(graph["out_idx"]).tolist()],
            "subgraph": as_list("kept", int),
        }
        self.graph_receipt = cf._jsonable(dict(graph.get("receipt") or {}))
        self._op_cache = None
        self._decode = None

    @property
    def edge_src(self) -> torch.Tensor:
        return self.col

    def spec(self) -> Dict[str, Any]:
        """JSON-safe: what :meth:`from_spec` needs (shapes + config) and the node anatomy."""
        return {"schema": LAB_SCHEMA, "config": self.cfg.to_dict(), "hidden_size": self.hidden_size, "n": self.n,
                "n_in": self.n_in, "n_out": self.n_out, "nnz": self.nnz, "steps": self.steps,
                "anatomy": self.anatomy, "graph_receipt": self.graph_receipt}

    def graph_meta(self) -> Dict[str, Any]:
        return {"n": self.n, "n_in": self.n_in, "n_out": self.n_out, "nnz": self.nnz, "steps": self.steps,
                "lab": self.spec()}

    @classmethod
    def from_spec(cls, hidden_size: int, spec: Dict[str, Any]) -> "ConnectomeLabCore":
        if spec.get("schema") != LAB_SCHEMA:
            raise ValueError(f"not a {LAB_SCHEMA} spec")
        core = cls(hidden_size, {k: int(spec[k]) for k in ("n", "n_in", "n_out", "nnz")},
                   LabConfig.from_dict(spec["config"]))
        core.anatomy = dict(spec.get("anatomy") or {})
        core.graph_receipt = dict(spec.get("graph_receipt") or {})
        return core

    # -- cache hygiene (same contract as FullConnectomeCore) ---------------------
    def _apply(self, fn, *args, **kwargs):
        self._op_cache = None
        return super()._apply(fn, *args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._op_cache = None
        self._decode = None
        return super()._load_from_state_dict(*args, **kwargs)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_op_cache"] = None
        state["_decode"] = None
        return state

    def reset_state(self) -> None:
        """Forget the stored decode state (a new conversation)."""
        self._decode = None

    # -- weights ---------------------------------------------------------------
    def edge_weights(self) -> torch.Tensor:
        """Current signed ``W`` values in CSR order (``bio_weight`` times the learned multiplier)."""
        base = self.bio_weight
        if self.cfg.edge_param == "residual":
            return base * (1.0 + self.edge_alpha)
        if self.cfg.edge_param == "dale_softplus":
            theta = self.edge_theta
            return base * (F.softplus(theta) / F.softplus(theta.new_tensor(THETA0)))
        return base

    def edge_multiplier(self) -> torch.Tensor:
        if self.cfg.edge_param == "residual":
            return 1.0 + self.edge_alpha
        if self.cfg.edge_param == "dale_softplus":
            return F.softplus(self.edge_theta) / F.softplus(self.edge_theta.new_tensor(THETA0))
        return torch.ones_like(self.bio_weight)

    def edge_parameters(self) -> List[nn.Parameter]:
        return [p for name, p in self.named_parameters() if name in ("edge_alpha", "edge_theta")]

    def _fixed_csr(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bufs = (self.crow, self.col, self.bio_weight, self.crow_t, self.col_t, self.perm_t)
        key = tuple((t.data_ptr(), t._version, str(t.device), t.dtype, t.numel()) for t in bufs)
        cache = self._op_cache
        if cache is None or cache[0] != key:
            w = cf.make_csr(self.crow, self.col, self.bio_weight, self.n)
            w_t = cf.make_csr(self.crow_t, self.col_t, self.bio_weight.index_select(0, self.perm_t), self.n)
            cache = (key, w, w_t)
            self._op_cache = cache
        return cache[1], cache[2]

    def _operator(self, dtype: torch.dtype) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Optional[Callable]]:
        """``(matmul, anchor)``: ``matmul(x) = W x``; ``anchor(zeros)`` (learned csr edges under
        autograd, else None) must be added to the first state of every recurrent chain."""
        cfg = self.cfg
        track = cfg.learned_edges and torch.is_grad_enabled() and any(p.requires_grad for p in self.edge_parameters())
        if cfg.backend == "csr_slab":
            if not cfg.learned_edges and dtype == self.bio_weight.dtype:
                w, w_t = self._fixed_csr()
                return (lambda x: cf.CSRMatmul.apply(x, w, w_t)), None
            values = self.edge_weights().to(dtype)
            vd = values.detach()
            w = cf.make_csr(self.crow, self.col, vd, self.n)
            w_t = cf.make_csr(self.crow_t, self.col_t, vd.index_select(0, self.perm_t), self.n)
            if not track:
                return (lambda x: cf.spmm(w, x)), None
            collector = EdgeGradCollector(w)
            return ((lambda x: _EdgeSpMM.apply(x, w, w_t, collector)),
                    (lambda zeros: _EdgeGradAnchor.apply(values, zeros, collector)))
        values = self.edge_weights().to(dtype)
        if not track:
            values = values.detach()
        idx = torch.stack([self.edge_dst.long(), self.col.long()])
        if cfg.backend == "torch_sparse":
            wsp = torch.sparse_coo_tensor(idx, values, (self.n, self.n), is_coalesced=True)
            return (lambda x: torch.sparse.mm(wsp, x)), None
        dense = values.new_zeros(self.n, self.n).index_put((idx[0], idx[1]), values)
        return (lambda x: dense @ x), None

    # -- dynamics ------------------------------------------------------------------
    def _coeffs(self, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {"a": torch.sigmoid(self.leak_logit).unsqueeze(1).to(dtype),
                "g_in": torch.exp(self.log_gain_in).unsqueeze(1).to(dtype),
                "g_out": torch.exp(self.log_gain_out).unsqueeze(1).to(dtype),
                "b": self.bias.unsqueeze(1).to(dtype)}

    def _run(self, u: torch.Tensor, r0: Optional[torch.Tensor], matmul, anchor, c: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``microsteps`` updates from state ``r0`` (None = exactly zero) with drive ``u`` (n, M)."""
        act = F.relu if self.cfg.activation == "relu" else torch.tanh
        a, g_in, g_out = c["a"], c["g_in"], c["g_out"]
        base = u + c["b"]
        r = r0
        for _ in range(self.steps):
            if r is None:  # r_0 = 0: W r_0 = 0, skip the spmm (the v3 path)
                r = a * act(base)
                if anchor is not None:
                    r = anchor(torch.zeros_like(r)) + r
            else:
                drive = g_in * matmul(g_out * r) + base
                r = (1 - a) * r + a * act(drive)
        return r

    def _gate(self, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.gate == "dynamic":
            return self.gate + torch.tanh(self.gate_proj(self.gate_norm(h)))
        return self.gate

    def _closed(self) -> bool:
        if bool(self.gate.detach().any()):
            return False
        return not (self.cfg.gate == "dynamic" and bool(self.gate_proj.weight.detach().any()))

    def forward(self, hidden: torch.Tensor, token_mask: Optional[torch.Tensor] = None,
                past_length: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, t, h = hidden.shape
        mask = (torch.ones(b, t, dtype=torch.bool, device=hidden.device) if token_mask is None
                else token_mask.to(dtype=torch.bool, device=hidden.device).reshape(b, t))
        info: Dict[str, Any] = {"M": int(mask.sum())}
        if info["M"] == 0:
            return hidden.clone(), info
        if not torch.is_grad_enabled() and self._closed():
            self._decode = None
            info["skipped"] = True  # closed gate: the write is exactly zero
            return hidden.clone(), info
        if self.cfg.persistent:
            return self._forward_scan(hidden, mask, past_length, info)
        return self._forward_reset(hidden, mask, info)

    def _forward_reset(self, hidden, mask, info):
        b, t, h = hidden.shape
        flat = hidden.reshape(b * t, h)
        sel = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        m = int(sel.numel())
        h_sel = flat.index_select(0, sel)
        u_in = self.in_proj(self.in_norm(h_sel))
        dtype = self.bio_weight.dtype
        matmul, anchor = self._operator(dtype)
        u = u_in.new_zeros(self.n, m, dtype=dtype).index_copy(0, self.in_idx, u_in.t().to(dtype))
        r = self._run(u, None, matmul, anchor, self._coeffs(dtype))
        e = r.index_select(0, self.out_idx).t()
        e_rms = torch.sqrt(e.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        write = self.out_proj((e / e_rms).to(hidden.dtype))
        g = self._gate(h_sel)
        new = h_sel + g * write
        out = flat.index_copy(0, sel, new).view(b, t, h)
        if not torch.is_grad_enabled():
            rows = sel // t
            same = (rows[1:] == rows[:-1]) & (sel[1:] == sel[:-1] + 1)
            info.update(self._stats(r, e, g, rows, b, pairs=(r[:, :-1][:, same], r[:, 1:][:, same]),
                                    write=write, hidden=h_sel))
        return out, info

    def _forward_scan(self, hidden, mask, past_length, info):
        b, t, h = hidden.shape
        dtype = self.bio_weight.dtype
        u_all = self.in_proj(self.in_norm(hidden)).to(dtype)
        matmul, anchor = self._operator(dtype)
        c = self._coeffs(dtype)
        rho = torch.sigmoid(self.retain_logit).unsqueeze(1).to(dtype) if self.cfg.state == "learned_decay" else None
        window = self.cfg.window
        start = 0
        r: Optional[torch.Tensor] = None
        if past_length is not None and int(past_length) > 0:
            st = self._decode
            if st is None or st["pos"] != int(past_length) or st["r"].shape[1] != b:
                raise RuntimeError("persistent CNS state missing or out of step for this cached decode step "
                                   f"(stored {None if st is None else st['pos']}, asked {past_length}); "
                                   "run the prompt with past_length=0 first")
            r, start = st["r"], int(past_length)
        es: List[torch.Tensor] = []
        cos_sum = torch.zeros((), dtype=dtype)
        cos_n = 0
        collect = not torch.is_grad_enabled()
        prev_new: Optional[torch.Tensor] = None
        prev_m: Optional[torch.Tensor] = None
        for i in range(t):
            pos = start + i
            if r is None or (window and pos % window == 0):
                r_start = None
            elif rho is not None:
                r_start = rho * r
            else:
                r_start = r
            u = u_all.new_zeros(self.n, b).index_copy(0, self.in_idx, u_all[:, i].t())
            r_new = self._run(u, r_start, matmul, anchor, c)
            mi = mask[:, i].unsqueeze(0)
            r = torch.where(mi, r_new, r if r is not None else torch.zeros_like(r_new))
            es.append(r_new.index_select(0, self.out_idx).t())
            if collect:
                if prev_new is not None:
                    both = (mask[:, i] & prev_m)
                    if bool(both.any()):
                        cs = F.cosine_similarity(prev_new[:, both], r_new[:, both], dim=0)
                        cos_sum = cos_sum + cs.sum()
                        cos_n += int(both.sum())
                prev_new, prev_m = r_new, mask[:, i]
        e = torch.stack(es, 1)  # (b, t, n_out)
        e_rms = torch.sqrt(e.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        write = self.out_proj((e / e_rms).to(hidden.dtype))
        g = self._gate(hidden)
        out = torch.where(mask.unsqueeze(-1), hidden + g * write, hidden)
        if past_length is not None and not torch.is_grad_enabled():
            self._decode = {"r": r.detach(), "pos": start + t}
        if collect:
            sel = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            g_sel = g.reshape(-1, h).index_select(0, sel) if g.dim() == 3 else g
            e_sel = e.reshape(b * t, -1).index_select(0, sel)
            info.update(self._stats(r, e_sel, g_sel, sel // t, b, pairs=None,
                                    write=write.reshape(b * t, h).index_select(0, sel),
                                    hidden=hidden.reshape(b * t, h).index_select(0, sel)))
            info["state_autocorr"] = float(cos_sum / cos_n) if cos_n else None
        return out, info

    @torch.no_grad()
    def _stats(self, r, e, g, rows, b, pairs, write: Optional[torch.Tensor] = None,
               hidden: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Diagnostics of one no-grad forward (never part of the output). ``write`` (M, h): the
        ungated write ``out_proj(e / rms)`` of the M real tokens; ``hidden`` (M, h): their hidden
        state at the graft; ``g``: the gate, (h,) or (M, h)."""
        out: Dict[str, Any] = {"mean_rate": float(r.mean()), "efferent_rms": float(e.pow(2).mean().sqrt()),
                               "fraction_active": float((r > 0).float().mean())}
        # gate_row_mean (run_lab's ``gate_by_source``) is only informative for gate="dynamic": a static
        # gate is the same vector for every token, so every row (and source) gets the same mean |g|.
        ga = g.abs().mean(dim=-1)
        if ga.dim() == 0:
            out["gate_row_mean"] = [float(ga)] * b
        else:
            tot = torch.zeros(b, dtype=ga.dtype).index_add_(0, rows, ga)
            cnt = torch.zeros(b, dtype=ga.dtype).index_add_(0, rows, torch.ones_like(ga))
            out["gate_row_mean"] = (tot / cnt.clamp_min(1)).tolist()
        if pairs is not None and pairs[0].shape[1] > 0:
            out["state_autocorr"] = float(F.cosine_similarity(pairs[0], pairs[1], dim=0).mean())
        if write is not None and hidden is not None and write.shape[0] > 0:
            # what the core adds to the residual stream, in float64 (these are small numbers)
            gw = (g * write).double()
            power = gw.pow(2).mean()  # mean over tokens and channels
            h_power = hidden.double().pow(2).mean()
            # write_rms_ratio: RMS of the gated write / RMS of the hidden state it is added to
            out["write_rms_ratio"] = float(torch.sqrt(power / h_power)) if float(h_power) > 0 else None
            # write_token_var_frac: share of the write's power that varies across tokens (about its
            # token mean, over this forward's tokens); 0 = a constant write (a learned bias vector)
            var = (gw - gw.mean(dim=0, keepdim=True)).pow(2).mean()
            out["write_token_var_frac"] = float(var / power) if float(power) > 0 else None
            # the two values above are per forward (they depend on which tokens share a forward); these
            # sufficient statistics let a caller pool them over any set of forwards: pool_write_moments
            out["write_moments"] = {"n": int(gw.shape[0]), "sum": gw.sum(dim=0), "sq": float(gw.pow(2).sum()),
                                    "h_sq": float(hidden.double().pow(2).sum())}
        return out


def pool_write_moments(moments: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """``(write_rms_ratio, write_token_var_frac)`` over every token of the given forwards.

    ``moments`` are ``_stats``' ``write_moments`` (``n`` tokens, per-channel ``sum`` of the gated write
    ``gw``, ``sq`` = sum of gw^2, ``h_sq`` = sum of hidden^2). Pooled exactly, so the result does not
    depend on how tokens were split into forwards: ratio = sqrt(sum gw^2 / sum h^2); the variance
    about the mean write over all tokens is sum gw^2 - |sum gw|^2 / n, and var_frac is its share of
    sum gw^2. (None, None) without moments; a value is None when its denominator is 0.
    """
    ms = [m for m in moments if m and int(m["n"]) > 0]
    if not ms:
        return None, None
    n = sum(int(m["n"]) for m in ms)
    s = torch.stack([m["sum"].double() for m in ms]).sum(0)
    sq = math.fsum(float(m["sq"]) for m in ms)
    h_sq = math.fsum(float(m["h_sq"]) for m in ms)
    ratio = math.sqrt(sq / h_sq) if h_sq > 0 else None
    var_frac = min(1.0, max(0.0, (sq - float(s.pow(2).sum()) / n) / sq)) if sq > 0 else None
    return ratio, var_frac


# ---------------------------------------------------------------------------
# construction helpers
# ---------------------------------------------------------------------------
def graph_from_full_core(full) -> dict:
    """A graph dict from a ``FullConnectomeCore``'s buffers (no names; magnitudes stand in for counts)."""
    crow = full.crow.long().cpu().numpy()
    post = np.repeat(np.arange(full.n), np.diff(crow))
    return {"n": full.n, "post": post.astype(np.int32), "pre": full.col.cpu().numpy().astype(np.int32),
            "value": full.val.cpu().numpy().astype(np.float32), "sign": full.sign.cpu().numpy().astype(np.int8),
            "in_idx": full.in_idx.cpu().numpy(), "out_idx": full.out_idx.cpu().numpy(), "weight": None,
            "receipt": {"source": "FullConnectomeCore buffers"}}


def lab_from_full(full, cfg: Optional[LabConfig] = None, graph: Optional[dict] = None) -> ConnectomeLabCore:
    """A lab core carrying a trained ``FullConnectomeCore``'s parameters on the same graph.

    ``cfg`` may change the dynamics / edge parametrisation / gate (the learned
    forms start exact) but not the graph. ``graph`` (optional) supplies the
    anatomy; its values must equal the core's.
    """
    cfg = cfg or LabConfig()
    if (cfg.topology, cfg.sign_mode, cfg.strength_init, cfg.norm, cfg.subgraph) != ("real", "dale", "bio", "frac", None):
        raise ValueError("lab_from_full keeps the full core's graph: topology real, dale, bio, frac, no subgraph")
    g = graph_from_full_core(full)
    if graph is not None:
        gv = np.asarray(graph["value"], dtype=np.float32)
        order = np.argsort(np.asarray(graph["post"], np.int64) * full.n + np.asarray(graph["pre"], np.int64), kind="stable")
        if not np.array_equal(gv[order], g["value"]):
            raise ValueError("graph values differ from the full core's buffers")
        g = graph
    core = ConnectomeLabCore(full.hidden_size, g, cfg)
    own = full.state_dict()
    with torch.no_grad():
        for name in ("in_norm.weight", "in_proj.weight", "out_proj.weight", "log_gain_in", "log_gain_out", "bias",
                     "leak_logit", "gate"):
            core.state_dict()[name].copy_(own[name])
    return core


def fresh_lab_core(hidden_size: int, graph: Optional[dict], cfg: LabConfig, seed: int) -> Optional[ConnectomeLabCore]:
    """A new core, deterministically initialised by ``seed`` (gate 0: the graft is the identity at birth)."""
    if cfg.topology == "none" or graph is None:
        return None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ConnectomeLabCore(hidden_size, graph, cfg)


# ---------------------------------------------------------------------------
# core files (lab runs) and anatomy export
# ---------------------------------------------------------------------------
def save_core_file(path: os.PathLike, core: ConnectomeLabCore, extra: Optional[Dict[str, Any]] = None,
                   keep_constant: bool = True) -> None:
    """Atomic save of one lab core. ``keep_constant=False`` drops the graph buffers; :func:`load_core_file`
    rebuilds them from the npz + config + seed and checks the array hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in core.state_dict().items()
          if keep_constant or k not in ConnectomeLabCore.CONST_BUFFERS}
    payload = {"schema": CORE_FILE_SCHEMA, "spec": core.spec(), "state_dict": sd, "extra": cf._jsonable(extra or {}),
               "constant_buffers": bool(keep_constant)}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_core_file(path: os.PathLike, cache_dir: Optional[os.PathLike] = None) -> Tuple[ConnectomeLabCore, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CORE_FILE_SCHEMA:
        raise ValueError(f"{path} is not a {CORE_FILE_SCHEMA} file")
    spec = payload["spec"]
    core = ConnectomeLabCore.from_spec(int(spec["hidden_size"]), spec)
    if not payload.get("constant_buffers", True):
        rec = spec.get("graph_receipt") or {}
        cfg = LabConfig.from_dict(spec["config"])
        g, _ = load_or_build_lab_graph(cfg, int(rec.get("graph_seed") or 0), cache_dir=cache_dir)
        if g["receipt"]["arrays_sha256"] != rec.get("arrays_sha256"):
            raise RuntimeError("rebuilt graph differs from the one this core was trained on")
        core.set_graph(g)
    missing, unexpected = core.load_state_dict(payload["state_dict"], strict=False)
    bad = [k for k in missing if payload.get("constant_buffers", True) or k not in ConnectomeLabCore.CONST_BUFFERS]
    if bad or unexpected:
        raise RuntimeError(f"core state mismatch: missing {bad[:5]} unexpected {list(unexpected)[:5]}")
    core.anatomy = dict(spec.get("anatomy") or core.anatomy)
    return core, payload


@torch.no_grad()
def export_anatomy(core: ConnectomeLabCore) -> Dict[str, np.ndarray]:
    """Per-edge and per-node arrays (numpy, no pickles): the anatomical map plus what training changed."""
    bio = core.bio_weight.detach().float().cpu().numpy()
    learned = core.edge_weights().detach().float().cpu().numpy()
    an = core.anatomy or {}
    out: Dict[str, np.ndarray] = {
        "edge_src": core.col.cpu().numpy().astype(np.int32),
        "edge_dst": core.edge_dst.cpu().numpy().astype(np.int32),
        "synapse_count": core.synapse_count.cpu().numpy().astype(np.int32),
        "biological_weight": bio,
        "learned_weight": learned,
        "learned_multiplier": np.divide(learned, bio, out=np.full_like(bio, np.nan), where=bio != 0),
        "input_nodes": core.in_idx.cpu().numpy(), "output_nodes": core.out_idx.cpu().numpy(),
        "leak": torch.sigmoid(core.leak_logit).cpu().numpy(), "gain_in": torch.exp(core.log_gain_in).cpu().numpy(),
        "gain_out": torch.exp(core.log_gain_out).cpu().numpy(), "bias": core.bias.detach().cpu().numpy(),
        "type_sign": core.type_sign.cpu().numpy(),
    }
    if hasattr(core, "retain_logit"):
        out["retain"] = torch.sigmoid(core.retain_logit).cpu().numpy()
    for key in ("node_id", "superclass", "nt"):
        if an.get(key) is not None:
            out[key] = np.asarray(an[key], dtype=str)
    if an.get("n_neurons") is not None:
        out["n_neurons"] = np.asarray(an["n_neurons"], dtype=np.int64)
    if an.get("subgraph") is not None:
        out["subgraph"] = np.asarray(an["subgraph"], dtype=np.int64)
    return out


def _core_from_path(path: str) -> ConnectomeLabCore:
    head = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    schema = head.get("schema")
    del head
    if schema == CORE_FILE_SCHEMA:
        return load_core_file(path)[0]
    import expanse_core as ec  # noqa: WPS433

    model, _, _ = ec.load_expanse(path)
    if not isinstance(model.cns_full, ConnectomeLabCore):
        raise ValueError(f"{path}: its CNS core is not a lab core ({type(model.cns_full).__name__})")
    return model.cns_full


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write the anatomical map + learned weights of a lab core to .npz")
    ex.add_argument("checkpoint")
    ex.add_argument("out")
    gr = sub.add_parser("graph", help="build one arm's graph and print its receipt")
    for f in dataclasses.fields(LabConfig):
        if f.name in ("topology", "sign_mode", "strength_init", "norm"):
            gr.add_argument(f"--{f.name}", default=f.default)
    gr.add_argument("--subgraph", type=int, default=None)
    gr.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "export":
        arrays = export_anatomy(_core_from_path(args.checkpoint))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **arrays)
        m = arrays["learned_multiplier"]
        print(json.dumps({"out": args.out, "edges": int(len(m)), "multiplier_mean": float(np.nanmean(m)),
                          "multiplier_min": float(np.nanmin(m)), "multiplier_max": float(np.nanmax(m))}, indent=1))
        return 0
    cfg = LabConfig(topology=args.topology, sign_mode=args.sign_mode, strength_init=args.strength_init, norm=args.norm,
                    subgraph=args.subgraph)
    g = lab_graph(cf.build_full_graph(npz_path()), cfg, args.seed)
    print(json.dumps(cf._jsonable(g["receipt"] if g else {"topology": "none"}), indent=1))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raise SystemExit(main())
