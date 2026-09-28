"""Sampling fairness, real exposure, failure-closed resume, and optimizer replay."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from source_sampler import SourceSampler, parse_source_weights, validate_resume_contract
import train_expanse as te


def make(mode="balanced", seed=29, weights=None):
    return SourceSampler(["large"] * 17 + ["small"] * 3, list(range(1, 21)),
                         mode=mode, seed=seed, weights=weights)


@pytest.mark.parametrize("batch", [1, 4, 7, 25])
def test_natural_is_byte_exact_legacy_order_and_tail_behavior(batch):
    sampler = make("natural")
    generator = torch.Generator().manual_seed(29)
    order, ptr = torch.randperm(20, generator=generator), 0
    for _ in range(60):
        if ptr + batch > len(order):
            order, ptr = torch.randperm(20, generator=generator), 0
        expected = order[ptr:ptr + batch]
        ptr += batch
        assert torch.equal(sampler.next_batch(batch), expected)
    assert torch.equal(sampler.generator.get_state(), generator.get_state())


def test_balanced_controls_row_probability_not_pool_or_token_probability():
    sampler = make(weights={"large": 1, "small": 3})
    indices = sampler.next_batch(12000)
    sampler.record_update(indices)
    exposure = sampler.receipt()["exposure"]
    assert exposure["rows"]["small"] / 12000 == pytest.approx(.75, abs=.025)
    assert exposure["rows"]["small"] + exposure["rows"]["large"] == 12000
    assert exposure["supervised_tokens"]["small"] == sum(i + 1 for i in indices.tolist() if i >= 17)
    assert sampler.receipt()["source_row_probabilities"] == {"large": .25, "small": .75}
    assert exposure["supervised_tokens"]["small"] > .75 * sum(exposure["supervised_tokens"].values())


def test_each_source_exhausts_its_pool_before_repeating():
    sampler = make()
    indices = sampler.next_batch(200).tolist()
    for name, members in sampler.members.items():
        selected = [i for i in indices if sampler.sources[i] == name]
        size = len(members)
        for start in range(0, len(selected) - size + 1, size):
            assert sorted(selected[start:start + size]) == members.tolist()


def test_local_sampling_does_not_advance_global_rng():
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    sampler = make()
    sampler.next_batch(500)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("mode", ["natural", "balanced"])
def test_serialized_resume_replays_rows_exposure_and_optimizer_exactly(mode):
    torch.manual_seed(123)
    weights = torch.randn(3, requires_grad=True)
    opt = torch.optim.AdamW([weights], lr=.01)
    sampler = make(mode)

    def update(s, w, optimizer):
        indices = s.next_batch(7)
        features = torch.stack([indices.float() / 20, torch.ones(len(indices)), indices.float().square() / 400], 1)
        features = torch.nn.functional.dropout(features, p=.2, training=True)
        loss = ((features @ w - (indices % 3).float()) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s.record_update(indices)
        return indices

    for _ in range(9):
        update(sampler, weights, opt)
    saved = deepcopy({"sampler": sampler.state_dict(), "weights": weights.detach(), "optimizer": opt.state_dict(),
                      "rng": torch.get_rng_state()})
    expected = [update(sampler, weights, opt) for _ in range(17)]
    resumed = make(mode)
    resumed.load_state_dict(saved["sampler"])
    other = saved["weights"].clone().requires_grad_()
    other_opt = torch.optim.AdamW([other], lr=.01)
    other_opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    actual = [update(resumed, other, other_opt) for _ in range(17)]
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(weights, other)
    assert sampler.receipt() == resumed.receipt()


@pytest.mark.parametrize("change", ["seed", "weights", "source", "tokens", "mode"])
def test_resume_rejects_changed_sampler_definition(change):
    original = make()
    args = dict(sources=original.sources, supervised_tokens=original.tokens, mode="balanced", seed=29)
    if change == "seed":
        args["seed"] = 30
    elif change == "weights":
        args["weights"] = {"large": 2, "small": 1}
    elif change == "source":
        args["sources"] = ["large"] * 16 + ["small"] * 4
    elif change == "tokens":
        args["supervised_tokens"] = [1] * 20
    else:
        args["mode"] = "natural"
    with pytest.raises(ValueError, match="mismatch"):
        SourceSampler(**args).load_state_dict(original.state_dict())


@pytest.mark.parametrize("change", ["order", "cursor", "schema", "counter", "rng"])
def test_resume_rejects_corrupt_state(change):
    sampler = make()
    state = sampler.state_dict()
    if change == "order":
        state["orders"]["large"][0] = state["orders"]["large"][1]
    elif change == "cursor":
        state["positions"]["small"] = 4
    elif change == "schema":
        state["config"]["schema"] = "future-unknown"
    elif change == "counter":
        state["exposure"]["rows"]["small"] = -1
    else:
        state["generator"] = torch.tensor([1])
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)


def test_legacy_resume_preserves_order_but_marks_unknown_previous_exposure():
    sampler = make("natural")
    sampler.next_batch(7)
    saved = sampler.state_dict()
    legacy = {"idx": saved["orders"]["__all__"], "ptr": saved["positions"]["__all__"], "gen": saved["generator"]}
    restored = make("natural")
    restored.load_legacy_order(legacy, step=42)
    assert torch.equal(sampler.next_batch(7), restored.next_batch(7))
    assert restored.receipt()["exposure"]["complete"] is False
    assert restored.receipt()["exposure"]["since_step"] == 42
    assert sum(restored.receipt()["exposure"]["rows"].values()) == 0
    with pytest.raises(ValueError, match="natural"):
        make().load_legacy_order(legacy, step=42)


@pytest.mark.parametrize("value", ["", "a", "a=1,a=2", "a=nan", "a=inf", "a=-1", "a=0"])
def test_weight_parser_rejects_bad_input(value):
    with pytest.raises(ValueError):
        parse_source_weights(value)


def test_weight_schema_requires_all_and_only_available_sources():
    assert parse_source_weights(" large=2, small=1 ") == {"large": 2, "small": 1}
    for weights in ({"large": 2}, {"large": 2, "small": 1, "unknown": 1}):
        with pytest.raises(ValueError, match="every"):
            make(weights=weights)
    with pytest.raises(ValueError, match="balanced"):
        make("natural", weights={"large": 1, "small": 1})


def test_resume_contract_and_full_metadata_identity():
    rows = [{"user": "q", "assistant": "a", "source": "a", "probs": [.1, .9]}]
    original = te.packed_data_fingerprint(rows, [], "legacy")
    changed = deepcopy(rows)
    changed[0]["probs"] = [.2, .8]
    assert te.packed_data_fingerprint(changed, [], "legacy") != original
    contract = {"parent_sha256": "a", "data_fingerprint": original, "arguments": {"steps": 80}}
    validate_resume_contract(deepcopy(contract), contract)
    for key in contract:
        changed_contract = deepcopy(contract)
        changed_contract[key] = "changed"
        with pytest.raises(ValueError, match=key):
            validate_resume_contract(changed_contract, contract)


def test_preserve_donor_wake_never_requests_a_reset_and_legacy_schedule_unchanged():
    schedule = te.parse_wake_schedule("preserve")
    assert all(te.donor_wake_progress(schedule, step, 80) is None for step in range(81))
    legacy = te.parse_wake_schedule("0.1,0.5")
    assert te.donor_wake_progress(legacy, 0, 100) is None
    assert te.donor_wake_progress(legacy, 10, 100) == 0
    assert te.donor_wake_progress(legacy, 30, 100) == pytest.approx(.5)
    assert te.donor_wake_progress(legacy, 90, 100) == 1
    with pytest.raises(ValueError):
        te.parse_wake_schedule("0.7,0.1")
