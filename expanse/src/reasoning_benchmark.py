"""Frozen, solver-labelled semantic groups and conservative numeric scoring.

This is a small arithmetic/loop-trace diagnostic, not a general reasoning
benchmark. Corpus screening recognizes only the explicitly documented shapes
below; absence of a recognized overlap is not proof of unseen semantics.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "supermix-semantic-reasoning-v1"
SCORER = "strict-final-number-v2"
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
MARKER = (r"(?:(?:final\s+answer|answer|total|result|(?:final\s+)?value\s+of\s+x)\b"
          r"(?:\s+(?:is\s+)?|\s*[:=]\s*)|x\b(?:\s+is\s+|\s*[:=]\s*))")
_SCALAR = re.compile(rf"^\s*({NUMBER})\s*[.!]?\s*$")
_TAIL = re.compile(rf"\b{MARKER}({NUMBER})\s*[.!]?\s*$", re.I)
_MARKED = re.compile(rf"\b{MARKER}({NUMBER})(?!\w|\.\d)", re.I)
_EXPR = re.compile(r"(?<![\w.])(-?\d+)\s*([+*/x-])\s*(-?\d+)(?![\w.])", re.I)
_INFIX = re.compile(r"(?<![\w.])-?\d+\s*[+*/x-]\s*(?=-?\d+)", re.I)
_AMBIGUOUS = re.compile(r"\b(?:not|wrong|incorrect|or|maybe|perhaps|possibly|approximately|either)\b", re.I)
_LOOP = re.compile(
    r"\bx\s*=\s*(-?\d+)\s*;?\s*for\s+i\s+in\s+range\s*\(\s*(\d+)\s*\)\s*:\s*"
    r"x\s*=\s*x\s*([+-])\s*(\d+)", re.I)


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def semantic_signature(prompt: str) -> tuple | None:
    """Recognize one binary integer expression or one literal add/subtract loop.

    This intentionally does not claim natural-language semantic equivalence.
    Unknown and compound expressions are not screened semantically.
    """
    loops = list(_LOOP.finditer(prompt))
    if len(loops) == 1:
        a, n, op, b = loops[0].groups()
        return ("python_trace", int(a), int(n), op, int(b))
    expressions = list(_EXPR.finditer(prompt))
    if len(expressions) != 1 or len(_INFIX.findall(prompt)) != 1 or re.search(r"\b(?:for|while|if)\b", prompt):
        return None
    a, op, b = expressions[0].groups()
    op = "*" if op.lower() == "x" else op
    a, b = int(a), int(b)
    if op in ("+", "*"):
        a, b = sorted((a, b))
    return ("arithmetic", a, op, b)


def corpus_index(paths: Iterable[Path]) -> dict:
    """Scan explicit JSONL inputs (all splits), hashing their exact bytes."""
    exact, semantic, files = set(), set(), []
    for path in sorted({Path(p).resolve() for p in paths}, key=str):
        before = file_sha256(path)
        rows = recognized = 0
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("user")
                if not isinstance(prompt, str):
                    raise ValueError(f"{path}:{number}: expected string user field")
                rows += 1
                exact.add(normalize_prompt(prompt))
                signature = semantic_signature(prompt)
                if signature is not None:
                    semantic.add(signature)
                    recognized += 1
        if file_sha256(path) != before:
            raise ValueError(f"Corpus changed while scanning: {path}")
        files.append({"path": str(path), "sha256": before, "rows": rows,
                      "recognized_semantic_rows": recognized})
    return {"exact": exact, "semantic": semantic, "files": files}


def _solve(family: str, a: int, b: int, op: str, n: int = 1) -> int:
    if family == "python_trace":
        return a + n * b if op == "+" else a - n * b
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/" and b and a % b == 0:
        return a // b
    raise ValueError("Unsupported or non-integral solver specification")


def make_group(family: str, a: int, b: int, op: str, n: int = 1) -> dict:
    spec = {"family": family, "a": a, "b": b, "op": op, "n": n}
    gid = canonical_sha256(spec)[:16]
    other = {"+": "-", "-": "+", "*": "/", "/": "*"}[op]
    # Division variants retain integral answers, so exact integer scoring is fair.
    delta = b if op == "/" else 1
    cases = [("base", a, op), ("numeric", a + delta, op), ("operation", a, other)]
    items = []
    for case, value, operator in cases:
        answer = _solve(family, value, b, operator, n)
        if family == "arithmetic":
            expr = f"{value} {operator} {b}"
            prompts = [f"What is {expr}?", f"Solve this basic math problem: {expr}"]
        elif family == "python_trace":
            code = f"x = {value}\nfor i in range({n}): x = x {operator} {b}"
            prompts = [f"{code}\nWhat is x after this runs?", f"What is the final value of x?\n{code}"]
        else:
            raise ValueError(f"Unknown family: {family}")
        for paraphrase, prompt in enumerate(prompts):
            items.append({"id": f"{gid}.{case}.{paraphrase}", "group_id": gid,
                          "family": family, "case": case, "paraphrase": paraphrase,
                          "user": prompt, "expected": answer})
    if len({i["expected"] for i in items if i["paraphrase"] == 0}) != 3:
        raise ValueError("Contrast answers must differ")
    return {"id": gid, "spec": spec, "items": items}


def make_suite(paths: Iterable[Path], *, seed: int = 20260927, groups_per_family: int = 6) -> dict:
    if groups_per_family < 1:
        raise ValueError("groups_per_family must be positive")
    index = corpus_index(paths)
    rng = random.Random(seed)
    groups, seen, excluded = [], set(), Counter()
    for family in ("arithmetic", "python_trace"):
        accepted = 0
        for attempt in range(10000):
            a, b, n = rng.randint(13, 97), rng.randint(2, 9), rng.randint(2, 7)
            op = "+" if family == "python_trace" or accepted % 2 == 0 else "/"
            if op == "/":
                a *= b
            group = make_group(family, a, b, op, n if family == "python_trace" else 1)
            signatures = {semantic_signature(i["user"]) for i in group["items"]}
            if group["id"] in seen:
                continue
            if any(normalize_prompt(i["user"]) in index["exact"] for i in group["items"]):
                excluded["normalized_exact_group_overlap"] += 1
                continue
            if signatures & index["semantic"]:
                excluded["recognized_semantic_group_overlap"] += 1
                continue
            if signatures & seen:
                continue
            seen.add(group["id"])
            seen.update(signatures)
            groups.append(group)
            accepted += 1
            if accepted == groups_per_family:
                break
        else:
            raise ValueError(f"Could not build enough disjoint groups for {family}")
    audit = {"files": index["files"], "excluded_groups": dict(excluded),
             "scope": "All rows of listed local JSONL files, including train/dev/heldout where present.",
             "semantic_recognizer": "One binary integer expression or literal x/for-range add/subtract loop.",
             "limitations": "Not a full training-history audit; unknown phrasings, upstream donor data and generated FlyCore rows are outside coverage. No claim of globally unseen semantics."}
    suite = {"schema": SCHEMA, "scorer": SCORER, "seed": seed,
             "groups_per_family": groups_per_family, "groups": groups, "corpus_audit": audit}
    return {**suite, "sha256": canonical_sha256(suite)}


def validate_suite(suite: dict) -> None:
    if suite.get("schema") != SCHEMA or suite.get("scorer") != SCORER:
        raise ValueError("Unknown suite/scorer schema")
    if suite.get("sha256") != canonical_sha256({k: v for k, v in suite.items() if k != "sha256"}):
        raise ValueError("Suite SHA-256 mismatch")
    groups = suite.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Suite has no groups")
    ids = set()
    for group in groups:
        if group != make_group(**group["spec"]) or group["id"] in ids:
            raise ValueError("Invalid solver-labelled group")
        ids.add(group["id"])
    families = Counter(g["spec"]["family"] for g in groups)
    if families != {"arithmetic": suite["groups_per_family"], "python_trace": suite["groups_per_family"]}:
        raise ValueError("Incomplete family group counts")


def verify_corpus(suite: dict) -> None:
    for entry in suite["corpus_audit"]["files"]:
        if file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Corpus bytes changed since suite freeze: {entry['path']}")


def load_suite(path: Path) -> dict:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_suite(suite)
    return suite


def select_groups(suite: dict, limit: int = 0) -> list[dict]:
    """A deterministic prefix round-robined across hash-sorted families.

    Limit is in groups, never individual prompts. A positive limit must retain
    both families. Group ids, including selection order, are recorded in reports.
    """
    validate_suite(suite)
    if limit < 0 or limit == 1:
        raise ValueError("limit must be zero (all) or at least two groups")
    families = defaultdict(list)
    for group in suite["groups"]:
        families[group["spec"]["family"]].append(group)
    ordered = []
    banks = [sorted(families[f], key=lambda g: canonical_sha256(g["id"])) for f in sorted(families)]
    for i in range(max(map(len, banks))):
        ordered.extend(bank[i] for bank in banks if i < len(bank))
    return ordered[:limit] if limit else ordered


def extract_final_number(text: str) -> tuple[Decimal | None, str]:
    """Accept a scalar or explicit terminal answer marker, never a number anywhere.

    Multiple conflicting explicit answer markers, trailing qualifications,
    alternative answers and unlabelled reasoning all fail closed. Intermediate
    arithmetic is not verified; final-answer correctness is the sole claim.
    Conservatively, differing intermediate ``x = ...`` / ``x is ...`` states
    count as conflicting markers too, even when a valid trace explains them.
    """
    text = text.strip()
    if _AMBIGUOUS.search(text):
        return None, "ambiguous_or_negated_answer"
    match = _SCALAR.fullmatch(text)
    if match:
        try:
            value = Decimal(match[1])
            return (value, "scalar") if value.is_finite() else (None, "nonfinite")
        except InvalidOperation:
            return None, "invalid_number"
    match = _TAIL.search(text)
    if not match:
        return None, "no_unambiguous_final_number"
    try:
        answers = {Decimal(m[1]) for m in _MARKED.finditer(text)}
        value = Decimal(match[1])
    except InvalidOperation:
        return None, "invalid_number"
    if len(answers) != 1 or value not in answers:
        return None, "conflicting_answer_markers"
    if not value.is_finite():
        return None, "nonfinite"
    return value, "explicit_final_number"


def score_reply(text: str, expected: int) -> dict:
    predicted, reason = extract_final_number(text)
    return {"predicted": str(predicted) if predicted is not None else None,
            "valid_answer": predicted is not None, "parse_reason": reason,
            "correct": predicted is not None and predicted == Decimal(expected)}


def summarize(groups: list[dict], results: list[dict]) -> dict:
    expected_items = [i for g in groups for i in g["items"]]
    by_id = {r["id"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {i["id"] for i in expected_items}:
        raise ValueError("Results must contain every selected item exactly once")
    families = defaultdict(list)
    group_correct = paraphrase_correct = consistent = changed = contrast_correct = 0
    for group in groups:
        rows = [by_id[i["id"]] for i in group["items"]]
        families[group["spec"]["family"]].extend(rows)
        group_correct += all(r["correct"] for r in rows)
        cases = {case: [by_id[i["id"]] for i in group["items"] if i["case"] == case]
                 for case in ("base", "numeric", "operation")}
        for pair in cases.values():
            paraphrase_correct += all(r["correct"] for r in pair)
            consistent += all(r["valid_answer"] for r in pair) and Decimal(pair[0]["predicted"]) == Decimal(pair[1]["predicted"])
        for case in ("numeric", "operation"):
            for j in (0, 1):
                base, altered = cases["base"][j], cases[case][j]
                changed += (base["valid_answer"] and altered["valid_answer"] and
                            Decimal(base["predicted"]) != Decimal(altered["predicted"]))
                contrast_correct += base["correct"] and altered["correct"]
    family_metrics = {f: {"n": len(rows), "correct": sum(r["correct"] for r in rows),
                          "accuracy": sum(r["correct"] for r in rows) / len(rows)} for f, rows in families.items()}
    return {"n": len(results), "correct": sum(r["correct"] for r in results),
            "accuracy": sum(r["correct"] for r in results) / len(results),
            "invalid_answers": sum(not r["valid_answer"] for r in results),
            "family_macro_accuracy": sum(v["accuracy"] for v in family_metrics.values()) / len(family_metrics),
            "families": family_metrics, "all_correct_groups": group_correct,
            "all_correct_group_rate": group_correct / len(groups),
            "paraphrase_pair_accuracy": paraphrase_correct / (3 * len(groups)),
            "paraphrase_consistency": consistent / (3 * len(groups)),
            "contrast_changed_answer_rate": changed / (4 * len(groups)),
            "contrast_pair_accuracy": contrast_correct / (4 * len(groups)),
            "interpretation": "Consistency or changed answers alone do not establish correctness. Final numeric answers only; intermediate reasoning is not verified."}
