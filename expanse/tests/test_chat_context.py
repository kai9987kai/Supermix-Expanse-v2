"""The chat must never silently answer a truncated, different question."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expanse_chat as chat


class Tokenizer:
    def __init__(self, length):
        self.length = length

    def encode_turn(self, prompt, reply):
        return list(range(self.length)), self.length

    def decode(self, ids):
        return " ".join(map(str, ids))


class Model:
    config = SimpleNamespace(max_position_embeddings=8)
    omni7 = None

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, x, **kwargs):
        self.calls.append((x.tolist(), kwargs["past_length"]))
        logits = torch.zeros(1, x.shape[1], 20)
        logits[..., 19] = 1
        return SimpleNamespace(logits=logits, past_key_values=())


def test_long_prompt_rejected_before_features_or_model(monkeypatch):
    monkeypatch.setattr(chat.ec.arch.OmniCore, "featurize", lambda _: (_ for _ in ()).throw(AssertionError()))
    model = Model()
    result = list(chat.stream_reply(model, Tokenizer(8), "whole question", 5, 0, True))
    assert "Please shorten" in result[0] and "8-token context" in result[0]
    assert model.calls == []


def test_context_budget_preserves_whole_prompt_and_skips_unused_last_forward(monkeypatch):
    monkeypatch.setattr(chat.ec.arch.OmniCore, "featurize", lambda _: torch.zeros(1, 1))
    model = Model()
    result = list(chat.stream_reply(model, Tokenizer(6), "whole question", 20, 0, True))
    assert len(result) == 2
    assert model.calls == [([[0, 1, 2, 3, 4, 5]], 0), ([[19]], 6)]


def test_requested_shorter_budget_is_respected(monkeypatch):
    monkeypatch.setattr(chat.ec.arch.OmniCore, "featurize", lambda _: torch.zeros(1, 1))
    model = Model()
    result = list(chat.stream_reply(model, Tokenizer(3), "whole question", 1, 0, True))
    assert len(result) == len(model.calls) == 1
