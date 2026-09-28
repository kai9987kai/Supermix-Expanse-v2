"""Content-bound inputs and ordered data receipts for Connectome Lab runs.

Fingerprints use canonical JSON and file bytes, never timestamps or Git status.
The file inventory is deliberately bounded to project code and selected inputs;
installed dependencies are described separately by the runner's code_version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List

INPUT_SCHEMA = "supermix-expanse-cns-lab-inputs-v1"
DATA_SCHEMA = "supermix-expanse-cns-lab-data-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(paths: Iterable[Path], project_root: Path) -> Dict[str, Any]:
    """Sorted, deduplicated file-byte identities relative to the project root.

    An explicit external checkpoint/NPZ on the same volume is represented with
    ``../`` components; no machine-specific absolute root enters the digest.
    Missing required inputs fail instead of masquerading as an empty corpus.
    """
    root = Path(project_root).resolve()
    files = {}
    for value in paths:
        path = Path(value).resolve()
        key = Path(os.path.relpath(path, root)).as_posix()
        if key not in files:
            files[key] = {"path": key, "sha256": file_sha256(path)}
    body = {"schema": INPUT_SCHEMA, "files": [files[key] for key in sorted(files)]}
    return {**body, "sha256": canonical_sha256(body)}


def project_source_files(project_root: Path) -> List[Path]:
    """Bounded source roots used by the model, tokenizer, grafts and trainer."""
    root = Path(project_root)
    expanse = root / "expanse"
    paths = list(expanse.glob("*.py"))
    for directory in (expanse / "src", expanse / "cns_lab",
                      root / "supermix-archimedes" / "archimedes" / "src",
                      root / "supermix-archimedes" / "models" / "supermix-v93" / "src"):
        paths.extend(directory.rglob("*.py"))
    # Vendored prompt encoder implementation (not site-packages). Keep this
    # bounded to its source directory rather than recursively walking a user
    # override that might also hold an environment or downloaded artifacts.
    omni = Path(os.environ.get("EXPANSE_OMNI_V7_DIR", str(root / "external" / "omni_v7")))
    paths.extend(omni.glob("*.py"))
    return paths


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _aggregate_problems(value: Dict[str, Any]) -> List[str]:
    if not _is_sha(value.get("sha256")):
        return ["invalid aggregate sha256"]
    try:
        expected = canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
    except (TypeError, ValueError):
        return ["manifest is not canonical JSON data"]
    return [] if value["sha256"] == expected else ["aggregate sha256 mismatch"]


def validate_input_manifest(value: Any) -> List[str]:
    """Validate the receipt itself, without consulting live files."""
    if not isinstance(value, dict):
        return ["input_manifest must be an object"]
    problems = []
    if value.get("schema") != INPUT_SCHEMA:
        problems.append("unknown input_manifest schema")
    if set(value) != {"schema", "files", "sha256"}:
        problems.append("unexpected input_manifest fields")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        problems.append("input_manifest files must be a nonempty list")
    else:
        names = []
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                problems.append("invalid input file entry")
                continue
            path = entry.get("path")
            if (not isinstance(path, str) or not path or "\\" in path or ":" in path
                    or PurePosixPath(path).is_absolute() or str(PurePosixPath(path)) != path):
                problems.append("input file path must be canonical and relative")
            else:
                names.append(path)
            if not _is_sha(entry.get("sha256")):
                problems.append("invalid input file sha256")
        if names != sorted(set(names)):
            problems.append("input file paths must be sorted and unique")
    return problems + _aggregate_problems(value)


def require_reusable_run(metrics: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Refuse unverifiable historical or stale completed caches; never edit them."""
    recorded = metrics.get("input_manifest")
    if recorded is None:
        raise RuntimeError("unverifiable-cache: completed run has no input_manifest (legacy run); "
                           "preserve it and use a new runs directory for a fresh run")
    problems = validate_input_manifest(recorded) + validate_input_manifest(current)
    if problems:
        raise RuntimeError("unverifiable-cache: " + "; ".join(problems))
    if recorded != current:
        raise RuntimeError("stale-cache: completed run input_manifest differs from current input bytes or "
                           "source files; preserve it and use a new runs directory for a fresh run")


def _row_receipt(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    # Include source/task metadata as well as text: they influence stratification
    # and exact-match scoring. List ordering and duplicate multiplicity matter.
    rows = list(rows)
    return {"count": len(rows), "sha256": canonical_sha256(rows)}


def data_lineage(data: Dict[str, Any], plan: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Membership/order after tokenization drops, plus the actual batch indices."""
    steps = [{key: (value.tolist() if hasattr(value, "tolist") else value)
              for key, value in step.items()} for step in plan]
    body = {"schema": DATA_SCHEMA, "train": _row_receipt(data["m_tr"]),
            "temporal_train": _row_receipt(data["m_tt"]),
            "eval": {key: _row_receipt(block["rows"]) for key, block in sorted(data["eval"].items())},
            "temporal_heldout": _row_receipt(data["temporal_heldout"]),
            "batch_plan": {"count": len(steps), "sha256": canonical_sha256(steps)}}
    return {**body, "sha256": canonical_sha256(body)}


def validate_data_lineage(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return ["data_lineage must be an object"]
    problems = []
    if value.get("schema") != DATA_SCHEMA:
        problems.append("unknown data_lineage schema")
    if set(value) != {"schema", "train", "temporal_train", "eval", "temporal_heldout", "batch_plan", "sha256"}:
        problems.append("unexpected data_lineage fields")
    blocks = [value.get(key) for key in ("train", "temporal_train", "temporal_heldout", "batch_plan")]
    evals = value.get("eval")
    if not isinstance(evals, dict) or not evals:
        problems.append("data_lineage eval must be a nonempty object")
    else:
        blocks.extend(evals.values())
    for block in blocks:
        if (not isinstance(block, dict) or set(block) != {"count", "sha256"}
                or type(block.get("count")) is not int or block["count"] < 0
                or not _is_sha(block.get("sha256"))):
            problems.append("invalid data_lineage row or batch receipt")
    return problems + _aggregate_problems(value)
