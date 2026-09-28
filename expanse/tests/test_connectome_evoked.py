"""Counterfactual readout and persistent-cache regressions on a tiny CPU graph."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import connectome_lab as cl  # noqa: E402

torch.set_num_threads(2)
HIDDEN = 6


def _graph(cfg):
    # Disjoint afferent/efferent sets, two complete input-output paths and
    # recurrent edges. Positive drive reaches both output types in four steps.
    pre = np.array([0, 1, 2, 3, 4, 5, 2, 3], dtype=np.int32)
    post = np.array([2, 3, 4, 5, 6, 7, 3, 2], dtype=np.int32)
    base = {"n": 8, "post": post, "pre": pre, "weight": np.ones(len(pre)),
            "value": np.ones(len(pre), dtype=np.float32), "sign": np.ones(8, dtype=np.int8),
            "in_idx": np.array([0, 1]), "out_idx": np.array([6, 7])}
    return cl.lab_graph(base, cfg, seed=11)


def _core(**overrides):
    cfg = cl.LabConfig(**{"readout": "evoked", "gate": "dynamic", "activation": "tanh", **overrides})
    core = cl.fresh_lab_core(HIDDEN, _graph(cfg), cfg, seed=13).double()
    gen = torch.Generator().manual_seed(19)
    with torch.no_grad():
        core.bias.copy_(torch.linspace(0.1, 0.4, core.n))
        core.gate.fill_(0.2)
        if cfg.gate == "dynamic":
            core.gate_proj.weight.copy_(torch.randn(core.gate_proj.weight.shape, generator=gen) * 0.1)
    return core


def _hidden(batch=2, tokens=5):
    return torch.randn(batch, tokens, HIDDEN, generator=torch.Generator().manual_seed(31), dtype=torch.float64)


def test_default_config_preserves_legacy_serialized_hash():
    cfg = cl.LabConfig()
    encoded = json.dumps(cfg.to_dict(), sort_keys=True).encode()
    assert "readout" not in cfg.to_dict()
    assert hashlib.sha256(encoded).hexdigest() == "ac4160fe2e03c45d90e3d2d9da4a14d7b0a8f3267e1cba9159bceb24a094a6f3"
    assert cl.LabConfig.from_dict(cfg.to_dict()).readout == "absolute"
    assert cl.LabConfig(readout="evoked").to_dict()["readout"] == "evoked"
    assert cl.LabConfig.from_dict(cl.LabConfig(readout="evoked").to_dict()).readout == "evoked"
    with pytest.raises(ValueError, match="readout"):
        cl.LabConfig(readout="not-a-mode")


@pytest.mark.parametrize("state", ["persist", "learned_decay", "window:3"])
def test_evoked_rejects_undefined_persistent_counterfactual(state):
    with pytest.raises(ValueError, match="requires state='reset'"):
        cl.LabConfig(readout="evoked", state=state)


def test_evoked_is_difference_of_complete_normalized_writes():
    core = _core(gate="static")
    absolute = copy.deepcopy(core)
    absolute.cfg = core.cfg.replace(readout="absolute")
    x = _hidden()
    zeros = torch.zeros(1, 1, HIDDEN, dtype=x.dtype)
    with torch.no_grad():
        expected = absolute(x)[0] - absolute(zeros)[0]
        actual = core(x)[0]
    assert torch.allclose(actual, expected, atol=2e-15, rtol=1e-13)
    assert float((actual - x).abs().max()) > 1e-4


@pytest.mark.parametrize("edge", ["residual", "dale_softplus"])
def test_evoked_sparse_gradients_match_dense_including_null_path(edge):
    sparse = _core(edge_param=edge)
    dense = copy.deepcopy(sparse)
    dense.cfg = dense.cfg.replace(backend="dense")
    x = _hidden()
    results = []
    for core in (sparse, dense):
        out, _ = core(x)
        (out.square().sum() + out.sin().sum()).backward()
        results.append(out.detach())
    assert torch.allclose(results[0], results[1], atol=1e-12, rtol=1e-12)
    for (name, a), (_, b) in zip(sparse.named_parameters(), dense.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        assert torch.allclose(a.grad, b.grad, atol=2e-11, rtol=1e-9), name
    assert float(sparse.in_proj.weight.grad.abs().max()) > 1e-5
    assert float(sparse.bias.grad.abs().max()) > 1e-5
    assert float(sparse.edge_parameters()[0].grad.abs().max()) > 1e-5


@pytest.mark.parametrize("case", ["zero_input", "zero_input_projection", "zero_edges", "empty"])
def test_evoked_cannot_write_through_bias_only_path(case):
    core = _core(topology="empty" if case == "empty" else "real", edge_param="dale_softplus")
    x = _hidden()
    with torch.no_grad():
        if case == "zero_input":
            x.zero_()
        elif case == "zero_input_projection":
            core.in_proj.weight.zero_()
        elif case == "zero_edges":
            core.bio_weight.zero_()
    # Biases, both gates and out_proj are nonzero; cancellation is the only
    # reason this core has no write. Check both training and inference paths.
    out, _ = core(x)
    assert torch.equal(out, x)
    if case in ("zero_edges", "empty"):
        out.square().sum().backward()
        for name, p in core.named_parameters():
            if p.grad is not None and p.numel():
                assert float(p.grad.abs().max()) < 1e-10, name
    with torch.no_grad():
        actual, info = core(x)
    assert torch.equal(actual, x)
    assert info["write_rms_ratio"] in (0.0, None)


def test_evoked_zero_gate_is_exact_identity_at_birth():
    cfg = cl.LabConfig(readout="evoked", gate="dynamic")
    core = cl.fresh_lab_core(HIDDEN, _graph(cfg), cfg, seed=7).double()
    x = _hidden()
    assert torch.equal(core(x)[0], x)
    with torch.no_grad():
        assert torch.equal(core(x)[0], x)


def test_evoked_is_token_local_batch_independent_and_preserves_pad():
    core = _core()
    x = _hidden()
    with torch.no_grad():
        expected = core(x)[0]
        individual = torch.cat([core(x[:, t:t + 1])[0] for t in range(x.shape[1])], dim=1)
        rows = torch.cat([core(x[b:b + 1])[0] for b in range(x.shape[0])], dim=0)
        changed = x.clone()
        changed[:, 3:] += 100.0
        changed_out = core(changed)[0]
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        mask[0, 1:3] = False
        masked_out = core(x, mask)[0]
    assert torch.allclose(individual, expected, atol=2e-15, rtol=1e-12)
    assert torch.allclose(rows, expected, atol=2e-15, rtol=1e-12)
    assert torch.equal(changed_out[:, :3], expected[:, :3])
    assert torch.equal(masked_out[~mask], x[~mask])
    assert torch.allclose(masked_out[mask], expected[mask], atol=2e-15, rtol=1e-12)


@pytest.mark.parametrize("readout", ["absolute", "evoked"])
def test_readout_core_file_roundtrip(tmp_path, readout):
    core = _core(readout=readout).float()
    path = tmp_path / "core.pt"
    cl.save_core_file(path, core)
    loaded, payload = cl.load_core_file(path)
    assert loaded.cfg.readout == readout
    assert ("readout" in payload["spec"]["config"]) == (readout == "evoked")
    x = _hidden().float()
    with torch.no_grad():
        assert torch.equal(core(x)[0], loaded(x)[0])


@pytest.mark.parametrize("state", ["persist", "learned_decay", "window:3"])
def test_pad_only_cached_chunks_advance_cursor_and_match_full_scan(state):
    core = _core(readout="absolute", state=state)
    x = _hidden(tokens=9)
    mask = torch.tensor([[False, False, True, False, False, False, True, True, True]]).expand(2, -1)
    with torch.no_grad():
        full = core(x, mask, past_length=0)[0]
        pieces = [core(x[:, :2], mask[:, :2], past_length=0)[0]]
        assert core._decode["pos"] == 2
        pieces.append(core(x[:, 2:3], mask[:, 2:3], past_length=2)[0])
        saved = core._decode["r"].clone()
        pieces.append(core(x[:, 3:6], mask[:, 3:6], past_length=3)[0])
        assert core._decode["pos"] == 6
        assert torch.equal(core._decode["r"], saved)
        pieces.append(core(x[:, 6:], mask[:, 6:], past_length=6)[0])
        assert torch.allclose(torch.cat(pieces, dim=1), full, atol=2e-15, rtol=1e-12)
        with pytest.raises(RuntimeError, match="out of step"):
            core(x[:, :2], mask[:, :2], past_length=1)


def test_module_dtype_change_invalidates_persistent_decode_state():
    core = _core(readout="absolute", state="persist").float()
    x = _hidden()
    with torch.no_grad():
        core(x[:, :2].float(), past_length=0)
        assert core._decode is not None
        core.double()
        assert core._decode is None
        with pytest.raises(RuntimeError, match="state missing"):
            core(x[:, 2:3], past_length=2)
        assert torch.isfinite(core(x, past_length=0)[0]).all()


@pytest.mark.parametrize("state", ["persist", "learned_decay", "window:3"])
def test_closed_gate_pad_only_cached_chunk_does_not_require_state(state):
    cfg = cl.LabConfig(state=state, gate="dynamic")
    core = cl.fresh_lab_core(HIDDEN, _graph(cfg), cfg, seed=7).double()
    x = _hidden()
    with torch.no_grad():
        assert torch.equal(core(x[:, :2], past_length=0)[0], x[:, :2])
        assert core._decode is None
        pads = torch.zeros(x.shape[0], 2, dtype=torch.bool)
        assert torch.equal(core(x[:, 2:4], pads, past_length=2)[0], x[:, 2:4])
        assert core._decode is None
        assert torch.equal(core(x[:, 4:], past_length=4)[0], x[:, 4:])


def test_evoked_core_cached_chunks_match_full_forward():
    core = _core()
    x = _hidden()
    with torch.no_grad():
        full = core(x, past_length=0)[0]
        pieces = [core(x[:, :2], past_length=0)[0]]
        pieces += [core(x[:, t:t + 1], past_length=t)[0] for t in range(2, x.shape[1])]
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=2e-15, rtol=1e-12)


def test_evoked_expanse_checkpoint_and_kv_decode_integration(tmp_path):
    """Exercise the actual model hook and checkpoint schema without a large donor checkpoint."""
    import expanse_core as ec

    cfg = ec.MiMoMixConfig(vocab_size=32, hidden_size=HIDDEN, n_layers=2, n_heads=1, n_kv_heads=1,
                          intermediate_size=12, use_moe=False, use_thinking_core=False, n_mtp_layers=0,
                          cns_after_layer=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        model = ec.ExpanseModel(cfg, with_omni=False).eval()
    ec.attach_lab_core(model, _core().float())
    tok = ec.text_utils.WordTokenizer([f"t{i}" for i in range(24)])
    path = tmp_path / "evoked_expanse.pt"
    ec.save_expanse(path, model, tok, {}, {"stage": "evoked-integration-test"})
    restored, _, _ = ec.load_expanse(path)
    assert restored.cns_full.cfg.readout == "evoked"
    x = torch.tensor([[6, 7, 8, 9, 10, 11]])
    kw = {"return_mtp": False, "use_fly": False}
    with torch.no_grad():
        expected = model(x, **kw).logits
        assert torch.equal(expected, restored(x, **kw).logits)
        assert not torch.allclose(expected, restored(x, use_cns_full=False, **kw).logits)
        out = restored(x[:, :3], use_cache=True, past_length=0, **kw)
        pieces, cache = [out.logits], out.past_key_values
        for pos in range(3, x.shape[1]):
            out = restored(x[:, pos:pos + 1], past_key_values=cache, use_cache=True, past_length=pos, **kw)
            pieces.append(out.logits)
            cache = out.past_key_values
        assert float((torch.cat(pieces, dim=1) - expected).abs().max()) < 1e-5
