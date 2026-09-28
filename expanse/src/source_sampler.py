"""Deterministic row sampling and measured training exposure.

Balanced weights describe source *row* probabilities, not token loss weights.
Rows within each selected source cycle through independently shuffled pools.
The natural mode reproduces the trainer's historical randperm and tail drop.
All randomness is local, and serialized state is tied to the packed row order.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import torch

SAMPLER_SCHEMA = "expanse-source-sampler-v1"


def parse_source_weights(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    weights = {}
    for entry in value.split(","):
        try:
            name, raw = entry.strip().split("=")
            name, weight = name.strip(), float(raw)
        except ValueError as exc:
            raise ValueError("source weights must be comma-separated source=weight pairs") from exc
        if not name or name in weights or not math.isfinite(weight) or weight <= 0:
            raise ValueError("source weights must have unique names and finite positive weights")
        weights[name] = weight
    return weights


class SourceSampler:
    def __init__(self, sources: Sequence[str], supervised_tokens: Sequence[int], *,
                 mode: str = "natural", seed: int = 2026,
                 weights: Mapping[str, float] | None = None):
        if mode not in ("natural", "balanced"):
            raise ValueError("sampling mode must be natural or balanced")
        self.sources = tuple(sources)
        self.tokens = tuple(int(n) for n in supervised_tokens)
        if not self.sources or len(self.sources) != len(self.tokens) or any(n < 0 for n in self.tokens):
            raise ValueError("sampler requires nonempty aligned source and token-count rows")
        self.names = tuple(sorted(set(self.sources)))
        self.members = {s: torch.tensor([i for i, name in enumerate(self.sources) if name == s], dtype=torch.long)
                        for s in self.names}
        if mode == "natural" and weights is not None:
            raise ValueError("source weights require balanced sampling")
        if weights is not None and set(weights) != set(self.names):
            raise ValueError("explicit source weights must name every nonempty packed source exactly; "
                             f"expected {','.join(self.names)}")
        raw = {s: float(weights[s]) if weights is not None else 1.0 for s in self.names}
        if any(not math.isfinite(w) or w <= 0 for w in raw.values()):
            raise ValueError("source weights must be finite and positive")
        total = sum(raw.values())
        if not math.isfinite(total):
            raise ValueError("source weight sum must be finite")
        self.probabilities = {s: (len(self.members[s]) / len(self.sources) if mode == "natural" else raw[s] / total)
                              for s in self.names}
        self.mode, self.seed = mode, int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.row_identity = hashlib.sha256(json.dumps(list(zip(self.sources, self.tokens)),
                                                       separators=(",", ":")).encode()).hexdigest()
        self.orders = {}
        self.positions = {}
        for s in ("__all__",) if mode == "natural" else self.names:
            self.orders[s] = self._permutation(s)
            self.positions[s] = 0
        self.rows_seen = {s: 0 for s in self.names}
        self.tokens_seen = {s: 0 for s in self.names}
        self.updates_seen = 0
        self.exposure_complete = True
        self.exposure_start_step = 0

    def _permutation(self, source):
        members = torch.arange(len(self.sources)) if source == "__all__" else self.members[source]
        return members[torch.randperm(len(members), generator=self.generator)]

    def config(self):
        return {"schema": SAMPLER_SCHEMA, "mode": self.mode, "seed": self.seed,
                "row_identity": self.row_identity, "source_row_probabilities": self.probabilities,
                "pool_rows": {s: len(self.members[s]) for s in self.names}}

    def next_batch(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.mode == "natural":
            name = "__all__"
            pos, order = self.positions[name], self.orders[name]
            if pos + batch_size > len(order):
                order = self.orders[name] = self._permutation(name)
                pos = 0
            result = order[pos:pos + batch_size].clone()
            # Keep the old trainer's behavior even when batch_size > pool size.
            self.positions[name] = pos + batch_size
            return result
        choices = torch.multinomial(torch.tensor([self.probabilities[s] for s in self.names], dtype=torch.float64),
                                    batch_size, replacement=True, generator=self.generator)
        result = []
        for choice in choices.tolist():
            name = self.names[choice]
            pos, order = self.positions[name], self.orders[name]
            if pos == len(order):
                order = self.orders[name] = self._permutation(name)
                pos = 0
            result.append(int(order[pos]))
            self.positions[name] = pos + 1
        return torch.tensor(result, dtype=torch.long)

    def record_update(self, indices: torch.Tensor) -> None:
        """Count exposure only after a successful optimizer update."""
        for i in indices.tolist():
            source = self.sources[i]
            self.rows_seen[source] += 1
            self.tokens_seen[source] += self.tokens[i]
        self.updates_seen += 1

    def receipt(self):
        return {**self.config(), "weight_unit": "row_probability",
                "exposure": {"complete": self.exposure_complete, "since_step": self.exposure_start_step,
                             "updates": self.updates_seen, "rows": dict(self.rows_seen),
                             "supervised_tokens": dict(self.tokens_seen),
                             "token_definition": "labels[:, 1:] != -100; counts repeated training exposures"}}

    def state_dict(self):
        return {"config": self.config(), "generator": self.generator.get_state().clone(),
                "orders": {s: order.clone() for s, order in self.orders.items()},
                "positions": dict(self.positions), "exposure": self.receipt()["exposure"]}

    def _validated_order(self, source, order):
        members = torch.arange(len(self.sources)) if source == "__all__" else self.members[source]
        if (not isinstance(order, torch.Tensor) or order.dtype != torch.long or order.ndim != 1
                or not torch.equal(order.cpu().sort().values, members.sort().values)):
            raise ValueError(f"invalid sampler permutation for {source}")
        return order.cpu().clone()

    def load_state_dict(self, state):
        if state.get("config") != self.config():
            raise ValueError("sampler state/config mismatch (mode, seed, weights, or packed row identity changed)")
        expected = set(self.orders)
        if set(state.get("orders", {})) != expected or set(state.get("positions", {})) != expected:
            raise ValueError("sampler pool state mismatch")
        orders, positions = {}, {}
        for source in expected:
            orders[source] = self._validated_order(source, state["orders"][source])
            pos = state["positions"][source]
            # Natural mode historically allows ptr > len when batch exceeds pool size.
            if type(pos) is not int or pos < 0 or (self.mode == "balanced" and pos > len(orders[source])):
                raise ValueError("invalid sampler cursor")
            positions[source] = pos
        exposure = state.get("exposure", {})
        for key in ("rows", "supervised_tokens"):
            values = exposure.get(key, {})
            if set(values) != set(self.names) or any(type(v) is not int or v < 0 for v in values.values()):
                raise ValueError("invalid sampler exposure counters")
        for key in ("updates", "since_step"):
            if type(exposure.get(key)) is not int or exposure[key] < 0:
                raise ValueError("invalid sampler exposure steps")
        if type(exposure.get("complete")) is not bool:
            raise ValueError("invalid sampler exposure completeness")
        generator = torch.Generator()
        try:
            generator.set_state(state["generator"])
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("invalid sampler generator state") from exc
        self.orders, self.positions, self.generator = orders, positions, generator
        self.rows_seen = dict(exposure["rows"])
        self.tokens_seen = dict(exposure["supervised_tokens"])
        self.updates_seen, self.exposure_start_step = exposure["updates"], exposure["since_step"]
        self.exposure_complete = exposure["complete"]

    def load_legacy_order(self, order, *, step: int):
        if self.mode != "natural":
            raise ValueError("legacy partials can only resume natural sampling")
        state = self.state_dict()
        state.update({"orders": {"__all__": order["idx"]}, "positions": {"__all__": order["ptr"]},
                      "generator": order["gen"]})
        state["exposure"].update({"complete": False, "since_step": step})
        self.load_state_dict(state)


def validate_resume_contract(saved, expected):
    """Training-affecting changes are new experiments, not exact resumes."""
    if saved != expected:
        keys = sorted(k for k in set(saved or {}) | set(expected) if (saved or {}).get(k) != expected.get(k))
        raise ValueError("resume contract mismatch: " + ", ".join(keys) + "; use a new output and --resume none")
