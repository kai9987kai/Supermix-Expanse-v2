"""Tests for the Connectome Laboratory core (``src/connectome_lab.py``).

What must hold, and why:

* v3 exactness   -> ``LabConfig()`` on a trained ``FullConnectomeCore`` is bitwise the
  same function (tiny graph and the real v3 checkpoint's core), and the learned-edge
  forms (``residual``, ``dale_softplus``) are exactly that function at birth;
* edge gradients -> the deferred ``sampled_addmm`` path (csr_slab) gives the same
  gradients as the dense backend's native autograd, for reset and persistent state;
* Dale           -> ``dale_softplus`` never flips an edge's sign, whatever the optimiser does;
* causality      -> for every state mode a change at token t leaves positions < t bitwise alone;
* decode         -> a cached decode (``past_length``) equals the full forward for persistent state;
* graph arms     -> shuffled keeps degrees / input multisets / signs, random keeps size,
  unsigned is all-positive, random strengths permute magnitudes, every arm has |W| radius 0.9;
* anatomy        -> export shapes, multiplier exactly 1 at birth;
* Expanse        -> a lab core round-trips through save_expanse / load_expanse and KV decode.

Run: ``python -m pytest tests/test_connectome_lab.py -q``
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import connectome_full as cf  # noqa: E402
import connectome_lab as cl  # noqa: E402
from test_connectome_full import write_tiny_npz  # noqa: E402

torch.set_num_threads(2)
V3 = ROOT / "checkpoints" / "supermix_expanse_v3.pt"
H = 24
STATES = ("reset", "persist", "window:3", "learned_decay")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    path = write_tiny_npz(tmp_path_factory.mktemp("lab") / "tiny.npz", n=48, density=0.2)
    return cf.build_full_graph(path)


def randomise(core, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in core.named_parameters():
            if name in ("edge_theta", "edge_alpha"):
                continue
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return core


def test_config_validation():
    assert cl.LabConfig().to_dict() == cl.LabConfig.from_dict(cl.LabConfig().to_dict()).to_dict()
    for bad in ({"topology": "x"}, {"state": "window:0"}, {"state": "window:k"}, {"microsteps": 0}, {"radius": 1.2}):
        with pytest.raises(ValueError):
            cl.LabConfig(**bad)
    with pytest.raises(ValueError):
        cl.LabConfig.from_dict({"nope": 1})
    assert cl.LabConfig(state="window:8").window == 8 and cl.LabConfig(state="persist").persistent


def test_default_lab_graph_is_the_v3_graph(base):
    g = cl.lab_graph(base, cl.LabConfig(), seed=0)
    assert np.array_equal(g["post"], base["post"]) and np.array_equal(g["pre"], base["pre"])
    assert np.array_equal(g["value"], base["value"])  # bitwise
    assert g["receipt"]["edge_overlap_with_real"] == 1.0


def test_lab_from_full_is_bitwise_v3(base):
    torch.manual_seed(0)
    full = randomise(cf.FullConnectomeCore(H, base, steps=4))
    x = torch.randn(2, 7, H)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    with torch.no_grad():
        ref, _ = full(x, mask)
    for edge in cl.EDGE_PARAMS:
        lab = cl.lab_from_full(full, cl.LabConfig(edge_param=edge), graph=base)
        with torch.no_grad():
            got, _ = lab(x, mask)
        assert torch.equal(got, ref), edge
        got_g, _ = lab(x, mask)  # under autograd (the anchor path for learned edges)
        assert torch.equal(got_g.detach(), ref), edge


def test_real_v3_core_exactness():
    if not V3.exists():
        pytest.skip("v3 checkpoint not present")
    head = torch.load(V3, map_location="cpu", weights_only=False, mmap=True)
    meta = head["expanse"]["cns"]
    sd = {k[len("cns_full."):]: v for k, v in head["state_dict"].items() if k.startswith("cns_full.")}
    hidden = int(head["config"]["hidden_size"])
    del head
    full = cf.FullConnectomeCore(hidden, {k: int(meta[k]) for k in ("n", "n_in", "n_out", "nnz")}, steps=int(meta["steps"]))
    full.load_state_dict(sd, strict=True)
    torch.set_num_threads(4)
    x = torch.randn(1, 24, hidden) * 3
    with torch.no_grad():
        ref, _ = full(x)
        for edge in ("fixed", "dale_softplus"):
            lab = cl.lab_from_full(full, cl.LabConfig(edge_param=edge))
            got, _ = lab(x)
            print(f"\n[v3 core] {edge}: max|d| {float((got - ref).abs().max()):.3e}")
            assert torch.equal(got, ref)
    torch.set_num_threads(2)


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("edge", ("residual", "dale_softplus"))
def test_edge_gradients_match_dense(base, state, edge):
    g = cl.lab_graph(base, cl.LabConfig(), seed=0)
    cfg = cl.LabConfig(edge_param=edge, state=state, microsteps=3, activation="tanh")
    torch.manual_seed(1)
    a = randomise(cl.ConnectomeLabCore(H, g, cfg), seed=3).double()
    with torch.no_grad():
        name = "edge_theta" if edge == "dale_softplus" else "edge_alpha"
        getattr(a, name).add_(torch.randn(a.nnz, dtype=torch.float64) * 0.2)
    b = copy.deepcopy(a)
    b.cfg = cfg.replace(backend="dense")
    x = torch.randn(2, 6, H, dtype=torch.float64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False
    outs = []
    for core in (a, b):
        out, _ = core(x, mask)
        (out.pow(2).sum() + out.sum()).backward()
        outs.append(out.detach())
    assert torch.allclose(outs[0], outs[1], atol=1e-12)
    ga, gb = getattr(a, name).grad, getattr(b, name).grad
    assert ga is not None and float(ga.abs().max()) > 0
    assert torch.allclose(ga, gb, rtol=1e-9, atol=1e-12), float((ga - gb).abs().max())
    assert torch.allclose(a.in_proj.weight.grad, b.in_proj.weight.grad, rtol=1e-9, atol=1e-12)


def test_torch_sparse_backend_matches(base):
    g = cl.lab_graph(base, cl.LabConfig(), seed=0)
    cfg = cl.LabConfig(edge_param="dale_softplus")
    a = randomise(cl.ConnectomeLabCore(H, g, cfg))
    b = copy.deepcopy(a)
    b.cfg = cfg.replace(backend="torch_sparse")
    x = torch.randn(2, 5, H)
    oa, _ = a(x)
    ob, _ = b(x)
    oa.sum().backward()
    ob.sum().backward()
    assert torch.allclose(oa, ob, atol=1e-5)
    assert torch.allclose(a.edge_theta.grad, b.edge_theta.grad, rtol=1e-4, atol=1e-7)


def test_dale_softplus_never_flips_sign(base):
    g = cl.lab_graph(base, cl.LabConfig(), seed=0)
    core = randomise(cl.ConnectomeLabCore(H, g, cl.LabConfig(edge_param="dale_softplus")))
    opt = torch.optim.SGD([core.edge_theta], lr=50.0)
    for _ in range(5):
        out, _ = core(torch.randn(2, 5, H))
        opt.zero_grad()
        (-out.pow(2).sum()).backward()
        opt.step()
    w = core.edge_weights().detach()
    assert float((core.edge_multiplier().detach() - 1).abs().max()) > 1e-3  # it did move
    assert torch.equal(torch.sign(w), torch.sign(core.bio_weight))


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("gate", ("static", "dynamic"))
def test_causal_for_every_state(base, state, gate):
    g = cl.lab_graph(base, cl.LabConfig(), seed=0)
    core = randomise(cl.ConnectomeLabCore(H, g, cl.LabConfig(state=state, gate=gate, edge_param="residual")), seed=5)
    x = torch.randn(1, 9, H)
    with torch.no_grad():
        y0, _ = core(x)
        x2 = x.clone()
        x2[:, 6] += 1.0
        y1, _ = core(x2)
    assert torch.equal(y0[:, :6], y1[:, :6])
    assert not torch.equal(y0[:, 6:], y1[:, 6:])


@pytest.mark.parametrize("state", ("persist", "window:4", "learned_decay"))
def test_persistent_decode_equals_full_forward(base, state):
    g = cl.lab_graph(base, cl.LabConfig(), seed=0)
    core = randomise(cl.ConnectomeLabCore(H, g, cl.LabConfig(state=state, gate="dynamic")), seed=7).eval()
    x = torch.randn(1, 11, H)
    with torch.no_grad():
        full, _ = core(x, past_length=0)
        steps = [core(x[:, :5], past_length=0)[0]]
        for t in range(5, 11):
            steps.append(core(x[:, t:t + 1], past_length=t)[0])
        inc = torch.cat(steps, 1)
        assert float((inc - full).abs().max()) < 1e-5
        with pytest.raises(RuntimeError):
            core(x[:, :1], past_length=3)  # out of step
    # the state really carries: the persistent core differs from a reset one on later tokens
    reset = copy.deepcopy(core)
    reset.cfg = core.cfg.replace(state="reset")
    with torch.no_grad():
        assert torch.equal(reset(x)[0][:, 0], core(x)[0][:, 0])
        assert not torch.allclose(reset(x)[0][:, 1:], core(x)[0][:, 1:])


def test_graph_arms(base):
    real = cl.lab_graph(base, cl.LabConfig(), 0)
    n = real["n"]
    shuf = cl.lab_graph(base, cl.LabConfig(topology="shuffled"), 3)
    for key in ("post", "pre"):
        assert np.array_equal(np.bincount(real[key], minlength=n), np.bincount(shuf[key], minlength=n))
    rows_r = np.bincount(real["post"], weights=np.abs(real["value"]), minlength=n)
    rows_s = np.bincount(shuf["post"], weights=np.abs(shuf["value"]), minlength=n)
    assert np.allclose(rows_r, rows_s, atol=1e-6)
    assert ((shuf["value"] > 0) == (real["sign"][shuf["pre"]] > 0)).all()
    assert shuf["receipt"]["edge_overlap_with_real"] < 0.9
    assert shuf["receipt"]["arrays_sha256"] != cl.lab_graph(base, cl.LabConfig(topology="shuffled"), 4)["receipt"]["arrays_sha256"]
    assert shuf["receipt"]["arrays_sha256"] == cl.lab_graph(base, cl.LabConfig(topology="shuffled"), 3)["receipt"]["arrays_sha256"]

    rnd = cl.lab_graph(base, cl.LabConfig(topology="random"), 1)
    assert rnd["n"] == n and len(rnd["post"]) == len(real["post"])
    assert len(np.unique(rnd["post"].astype(np.int64) * n + rnd["pre"])) == len(rnd["post"])

    uns = cl.lab_graph(base, cl.LabConfig(sign_mode="unsigned"), 0)
    assert (uns["value"] > 0).all()
    assert np.array_equal(np.abs(uns["value"]), np.abs(real["value"]))

    rs = cl.lab_graph(base, cl.LabConfig(strength_init="random"), 2)
    assert np.array_equal(rs["post"], real["post"])
    ratio = np.sort(np.abs(rs["value"]).astype(np.float64)) / np.sort(np.abs(real["value"]).astype(np.float64))
    assert np.allclose(ratio, ratio[0], rtol=1e-5)  # the same magnitudes, reassigned (and rescaled)

    for cfg in (cl.LabConfig(norm="sqrt_indeg"), cl.LabConfig(norm="raw"), cl.LabConfig(topology="random"),
                cl.LabConfig(strength_init="random")):
        g = cl.lab_graph(base, cfg, 0)
        assert abs(g["receipt"]["abs_spectral_radius"] - 0.9) < 1e-4, cfg
    for cfg in (cl.LabConfig(), cl.LabConfig(topology="shuffled")):  # row sums 0.9: Perron root <= 0.9
        g = cl.lab_graph(base, cfg, 0)
        assert g["receipt"]["abs_spectral_radius"] <= 0.9 + 1e-6
        rows = np.bincount(g["post"], weights=np.abs(g["value"]), minlength=g["n"])
        assert np.allclose(rows[rows > 0], 0.9, atol=1e-6)
    assert cl.lab_graph(base, cl.LabConfig(topology="none"), 0) is None


def test_subgraph(base):
    g = cl.lab_graph(base, cl.LabConfig(subgraph=20), 0)
    assert g["n"] == 20 and len(g["kept"]) == 20
    assert len(g["in_idx"]) >= 1 and len(g["out_idx"]) >= 1
    kept = g["kept"]
    orig = set(zip(base["post"].tolist(), base["pre"].tolist()))
    assert all((int(kept[p]), int(kept[q])) in orig for p, q in zip(g["post"], g["pre"]))
    assert [str(base["type_names"][k]) for k in kept] == [str(x) for x in g["type_names"]]
    core = cl.ConnectomeLabCore(H, g, cl.LabConfig(subgraph=20))
    assert core.anatomy["subgraph"] == [int(k) for k in kept]


@pytest.mark.parametrize("gate", ("static", "dynamic"))
def test_empty_topology_is_the_no_connectome_control(base, gate):
    """topology="empty": same types, no edges; the write is a constant (static gate) or a pure adapter (dynamic)."""
    cfg = cl.LabConfig(topology="empty", edge_param="dale_softplus", gate=gate)
    assert not cfg.graph_is_seeded() and list(cfg.to_dict()) == list(cl.LabConfig().to_dict())
    g = cl.lab_graph(base, cfg, 0)
    assert len(g["post"]) == 0 and g["n"] == base["n"] and len(g["in_idx"]) and len(g["out_idx"])
    core = randomise(cl.fresh_lab_core(H, g, cfg, seed=0))
    x = torch.randn(2, 6, H)
    out, _ = core(x)
    out.sum().backward()
    assert core.gate.grad is not None and float(core.gate.grad.abs().sum()) > 0
    with torch.no_grad():
        write = (core(x)[0] - x).reshape(-1, H)
    spread = float((write - write.mean(0)).abs().max())
    if gate == "static":
        assert spread < 1e-5  # the output types see only their bias
    else:
        assert spread > 1e-3 and float(core.gate_proj.weight.grad.abs().sum()) > 0
    assert cl.export_anatomy(core)["edge_src"].shape == (0,)


def test_anatomy_export_and_core_file(base, tmp_path):
    g = cl.lab_graph(base, cl.LabConfig(), 0)
    core = cl.fresh_lab_core(H, g, cl.LabConfig(edge_param="dale_softplus"), seed=4)
    again = cl.fresh_lab_core(H, g, cl.LabConfig(edge_param="dale_softplus"), seed=4)
    assert all(torch.equal(a, b) for a, b in zip(core.state_dict().values(), again.state_dict().values()))
    arrays = cl.export_anatomy(core)
    e = len(g["post"])
    for k in ("edge_src", "edge_dst", "synapse_count", "biological_weight", "learned_weight", "learned_multiplier"):
        assert arrays[k].shape == (e,), k
    assert (arrays["learned_multiplier"] == 1.0).all()
    assert arrays["synapse_count"].sum() == int(np.rint(g["weight"]).sum())
    assert list(arrays["node_id"]) == [str(x) for x in g["type_names"]]
    path = tmp_path / "core.pt"
    cl.save_core_file(path, core, {"note": "t"})
    back, _ = cl.load_core_file(path)
    assert all(torch.equal(a, b) for a, b in zip(core.state_dict().values(), back.state_dict().values()))


def test_expanse_roundtrip_and_decode(tmp_path):
    if not V3.exists():
        pytest.skip("v3 checkpoint not present")
    import expanse_core as ec

    torch.set_num_threads(4)
    model, tok, payload = ec.load_expanse(V3)
    payload["state_dict"] = None
    base = cf.build_full_graph(cl.npz_path())
    cfg = cl.LabConfig(subgraph=300, edge_param="dale_softplus", state="persist", gate="dynamic")
    g = cl.lab_graph(base, cfg, 0)
    core = randomise(cl.fresh_lab_core(model.config.hidden_size, g, cfg, seed=1), scale=0.05)
    ec.attach_lab_core(model, core)
    path = tmp_path / "lab.pt"
    ec.save_expanse(path, model, tok, {}, {"stage": "lab-test"})
    back, _, _ = ec.load_expanse(path)
    ids, plen = tok.encode_turn("How many cell types does the male CNS have?", "It has many types.")
    x = torch.tensor([ids])
    with torch.no_grad():
        ref = model(x, return_mtp=False).logits
        got = back(x, return_mtp=False).logits
        assert torch.equal(ref, got)
        assert back.cns_full.anatomy["node_id"] == core.anatomy["node_id"]
        out = back(x[:, :plen], use_cache=True, return_mtp=False, past_length=0)
        steps, past = [out.logits[0]], out.past_key_values
        for t in range(plen, x.shape[1]):
            out = back(x[:, t:t + 1], past_key_values=past, use_cache=True, return_mtp=False, past_length=t)
            past = out.past_key_values
            steps.append(out.logits[0])
        inc = torch.cat(steps, 0)
    d = float((inc - ref[0]).abs().max())
    print(f"\n[expanse lab decode] max|dlogit| {d:.3e}")
    assert d < 1e-4
    ec.attach_lab_core(model, None)
    assert model.expanse_spec()["cns"] is None
    torch.set_num_threads(2)
