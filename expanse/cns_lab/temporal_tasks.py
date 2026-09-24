"""Synthetic tasks that need memory across the prompt, for the persistent-state hypothesis.

Four families, each a short single-turn row whose answer depends on tokens
far back in the prompt (all well inside the 128-token context):

    recall    "Remember these codes: amber 7, cedar 2, ... Then: <filler>. What was the code for cedar?" -> "2"
    pattern   "Continue the sequence: 4 9 1 4 9 1 4" -> "9 1 4"
    count     "How many times does 'fox' appear? fox oak fox ..." -> "3"
    first     "Here is a list: stone, river, cloud, lamp. What was the first item?" -> "stone"

``make_rows(n, seed, split)`` is deterministic; ``train`` and ``heldout`` use
disjoint seed streams and ``split_rows`` removes any heldout prompt from the
train rows, so no evaluated prompt is ever trained on.
"""

from __future__ import annotations

import random
from typing import Dict, List

WORDS = ("amber cedar maple river stone cloud lamp tiger otter falcon copper silver garden harbor meadow canyon "
         "island lantern mirror pepper violet walnut willow zebra anchor basket candle desert ember forest glacier "
         "hammer jungle kettle ladder magnet needle orchard pillow quartz saddle thunder velvet wagon yarn apple "
         "bridge circle dragon engine feather ginger honey igloo jacket koala lemon mango nectar olive parrot").split()
FAMILIES = ("recall", "pattern", "count", "first")
SPLIT_SALT = {"train": 11, "heldout": 97}


def _recall(rng: random.Random) -> Dict[str, str]:
    keys = rng.sample(WORDS, rng.randint(4, 6))
    codes = [str(rng.randint(0, 9)) for _ in keys]
    filler = " ".join(rng.sample([w for w in WORDS if w not in keys], rng.randint(6, 12)))
    q = rng.randrange(len(keys))
    pairs = ", ".join(f"{k} {c}" for k, c in zip(keys, codes))
    return {"user": f"Remember these codes: {pairs}. Then: {filler}. What was the code for {keys[q]}?",
            "assistant": codes[q]}


def _pattern(rng: random.Random) -> Dict[str, str]:
    period = rng.randint(2, 4)
    unit = [str(rng.randint(0, 9)) for _ in range(period)]
    while len(set(unit)) == 1:
        unit = [str(rng.randint(0, 9)) for _ in range(period)]
    shown = rng.randint(2 * period + 1, 3 * period + period - 1)
    seq = [unit[i % period] for i in range(shown + period)]
    return {"user": "Continue the sequence: " + " ".join(seq[:shown]), "assistant": " ".join(seq[shown:])}


def _count(rng: random.Random) -> Dict[str, str]:
    target = rng.choice(WORDS)
    k = rng.randint(1, 5)
    others = [rng.choice([w for w in WORDS if w != target]) for _ in range(rng.randint(6, 11))]
    items = others + [target] * k
    rng.shuffle(items)
    return {"user": f"How many times does '{target}' appear? " + " ".join(items), "assistant": str(k)}


def _first(rng: random.Random) -> Dict[str, str]:
    items = rng.sample(WORDS, rng.randint(4, 8))
    return {"user": "Here is a list: " + ", ".join(items) + ". What was the first item?", "assistant": items[0]}


MAKERS = {"recall": _recall, "pattern": _pattern, "count": _count, "first": _first}


def make_rows(n: int, seed: int, split: str = "train") -> List[Dict[str, str]]:
    """``n`` rows, families round-robin, deterministic in (seed, split)."""
    rng = random.Random(seed * 1000 + SPLIT_SALT[split])
    rows = []
    for i in range(int(n)):
        fam = FAMILIES[i % len(FAMILIES)]
        r = MAKERS[fam](rng)
        rows.append({**r, "source": "temporal", "task": fam, "split": split})
    return rows


def split_rows(n_train: int, n_heldout: int, seed: int) -> Dict[str, List[Dict[str, str]]]:
    held = make_rows(n_heldout, seed, "heldout")
    seen = {r["user"] for r in held}
    train = [r for r in make_rows(n_train + n_heldout, seed, "train") if r["user"] not in seen][:n_train]
    return {"train": train, "heldout": held}


def exact_match(pred: str, gold: str) -> bool:
    return " ".join(pred.strip().rstrip(".").split()) == " ".join(gold.strip().split())
