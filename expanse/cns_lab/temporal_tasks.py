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

``split_rows(..., version=2)`` (protocol ``temporal.split_version``; default 1
is the original split, unchanged) also removes near-duplicates: every train row
whose user text *starts with* a held-out user text (a held-out ``pattern``
prompt that is a prefix of a train prompt shows the train row the held-out
answer), and every train ``pattern`` row whose repeating unit is a held-out
row's unit (compared up to rotation: the same cycle entered at another phase).
Version 2 keeps the family round-robin of the train rows and draws further
candidates from the same train stream when rows are dropped; the held-out
rows are identical in both versions.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

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


SPLIT_VERSIONS = (1, 2)
PATTERN_PREFIX = "Continue the sequence: "


def pattern_unit(row: Dict[str, str]) -> Optional[Tuple[str, ...]]:
    """The repeating unit of a ``pattern`` row (minimal period of prompt + answer), canonical up to
    rotation (the lexicographically smallest rotation); None for other rows."""
    if row.get("task") != "pattern" or not row["user"].startswith(PATTERN_PREFIX):
        return None
    seq = row["user"][len(PATTERN_PREFIX):].split() + row["assistant"].split()
    if not seq:
        return None
    period = next(p for p in range(1, len(seq) + 1) if all(seq[i] == seq[i % p] for i in range(len(seq))))
    unit = seq[:period]
    return min(tuple(unit[i:] + unit[:i]) for i in range(period))


def _leaks(row: Dict[str, str], held_users: List[str], held_units: set) -> bool:
    """Split v2: a train row too close to a held-out row."""
    user = row["user"]
    if any(user.startswith(hu) for hu in held_users):
        return True
    unit = pattern_unit(row)
    return unit is not None and unit in held_units


def split_rows(n_train: int, n_heldout: int, seed: int, version: int = 1) -> Dict[str, List[Dict[str, str]]]:
    """Train and held-out rows. ``version=1``: the original split (exact held-out prompts removed);
    ``version=2``: also drops near-duplicates of held-out rows (see module doc)."""
    version = int(version)
    if version not in SPLIT_VERSIONS:
        raise ValueError(f"temporal split_version {version}; allowed {SPLIT_VERSIONS}")
    held = make_rows(n_heldout, seed, "heldout")
    seen = {r["user"] for r in held}
    if version == 1:
        train = [r for r in make_rows(n_train + n_heldout, seed, "train") if r["user"] not in seen][:n_train]
        return {"train": train, "heldout": held}
    held_users = sorted(seen)
    held_units = {u for u in (pattern_unit(r) for r in held) if u is not None}
    n_train = int(n_train)
    pool = n_train + n_heldout
    while True:
        cands = make_rows(pool, seed, "train")  # a longer stream extends a shorter one (same rng sequence)
        queues: Dict[str, List[Dict[str, str]]] = {f: [] for f in FAMILIES}
        for r in cands:
            if r["user"] not in seen and not _leaks(r, held_users, held_units):
                queues[r["task"]].append(r)
        need = {f: n_train // len(FAMILIES) + (1 if i < n_train % len(FAMILIES) else 0) for i, f in enumerate(FAMILIES)}
        if all(len(queues[f]) >= need[f] for f in FAMILIES) or pool > 64 * (n_train + n_heldout + 1):
            break
        pool *= 2
    ptr = {f: 0 for f in FAMILIES}
    train: List[Dict[str, str]] = []
    i = 0
    while len(train) < n_train and any(ptr[f] < len(queues[f]) for f in FAMILIES):
        f = FAMILIES[i % len(FAMILIES)]
        if ptr[f] < len(queues[f]):
            train.append(queues[f][ptr[f]])
            ptr[f] += 1
        i += 1
    return {"train": train, "heldout": held}


def exact_match(pred: str, gold: str) -> bool:
    return " ".join(pred.strip().rstrip(".").split()) == " ".join(gold.strip().split())
