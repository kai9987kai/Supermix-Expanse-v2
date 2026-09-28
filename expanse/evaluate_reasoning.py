"""Freeze and evaluate a small semantic-group reasoning diagnostic.

    python expanse/evaluate_reasoning.py --make-suite expanse/checkpoints/reasoning-suite.json
    python expanse/evaluate_reasoning.py --suite expanse/checkpoints/reasoning-suite.json \
        --model parent=expanse/checkpoints/supermix_expanse_v4.pt --out report.json

Suite creation does not load a model. Evaluation loads named full checkpoints
sequentially, uses their native tokenizer/grafts and greedy KV-cache decoding,
and records every raw generation. No generated code is executed.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cns_lab"))
import reasoning_benchmark as rb
import lineage


def default_corpus_paths() -> list[Path]:
    """Bounded known file-backed training inputs; includes all their splits."""
    paths = [PROJECT / "supermix-archimedes" / "corpus" / f"{name}.jsonl" for name in ("omni", "code", "math")]
    paths += [ROOT / "data" / name for name in ("code_rows.jsonl", "bio_rows.clean.jsonl", "connectome_rows.jsonl")]
    paths += [ROOT / "data" / "v3" / name for name in
              ("fresh_omni.jsonl", "fresh_code.jsonl", "fresh_math.jsonl", "connectome_rows_v3.jsonl")]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Default corpus audit inputs missing; supply explicit --corpus paths: {missing}")
    return paths


def write_new_json(path: Path, value: dict) -> None:
    """Refuse to overwrite evidence; identical suite creation is idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) == value:
            return
        raise FileExistsError(f"Refusing to overwrite existing evidence: {path}")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def generation_budget(prompt_tokens: int, requested: int, context: int) -> int:
    if requested < 1 or context < 1 or prompt_tokens < 1:
        raise ValueError("Token budgets must be positive")
    if prompt_tokens >= context:
        raise ValueError(f"Prompt uses {prompt_tokens} tokens with context {context}; no answer budget")
    return min(requested, context - prompt_tokens)


def evaluate_one(name: str, checkpoint: Path, groups: list[dict], decode: dict) -> dict:
    import torch
    import expanse_core as ec
    import train_expanse as te

    checkpoint = checkpoint.resolve()
    before = rb.file_sha256(checkpoint)
    torch.manual_seed(decode["seed"])
    model, tok, payload, schema = te.load_checkpoint(checkpoint)
    model.eval()
    context = int(model.config.max_position_embeddings)
    tok_sha = rb.canonical_sha256(payload["tokenizer"])
    del payload
    results = []
    started = time.monotonic()
    items = [i for g in groups for i in g["items"]]
    with torch.inference_mode():
        for number, item in enumerate(items, 1):
            row = {**item, "raw_output": "", "token_ids": [], "error": None}
            stamp = time.monotonic()
            ids, _ = tok.encode_turn(item["user"], None)
            row["prompt_tokens"] = len(ids)
            row["context_tokens"] = context
            row["prompt_truncated"] = False
            row["budget_exhausted"] = False
            try:
                budget = generation_budget(len(ids), decode["max_new_tokens"], context)
                row["max_new_tokens_effective"] = budget
                kwargs = {"omni_features": ec.arch.OmniCore.featurize([item["user"]])}
                if model.omni7 is not None:
                    kwargs["omni7_state"] = model.omni7_state_for([item["user"]])
                new = ec.greedy_decode(model, torch.tensor([ids]), max_new_tokens=budget, **kwargs)
                row["token_ids"] = new
                row["raw_output"] = tok.decode(new).strip()
                row["budget_exhausted"] = len(new) >= budget and (not new or new[-1] != ec.text_utils.EOS)
                row.update(rb.score_reply(row["raw_output"], item["expected"]))
                if row["budget_exhausted"]:
                    row.update(valid_answer=False, correct=False, parse_reason="generation_budget_exhausted")
            except (ValueError, RuntimeError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row.update(predicted=None, valid_answer=False, correct=False, parse_reason="generation_error")
            row["elapsed_seconds"] = time.monotonic() - stamp
            results.append(row)
            if number % 6 == 0:
                print(f"[reasoning] {name}: {number}/{len(items)}; correct={sum(r['correct'] for r in results)}", flush=True)
    if rb.file_sha256(checkpoint) != before:
        raise ValueError(f"Checkpoint bytes changed during evaluation: {checkpoint}")
    result = {"name": name, "checkpoint": str(checkpoint), "checkpoint_sha256": before,
              "checkpoint_schema": schema, "tokenizer_sha256": tok_sha, "context_tokens": context,
              "elapsed_seconds": time.monotonic() - started, "metrics": rb.summarize(groups, results),
              "items": results}
    del model, tok
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-suite", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--corpus", action="append", type=Path, help="Explicit audit JSONL; repeat for each file")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--groups-per-family", type=int, default=6)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Whole groups; 0=all, otherwise at least 2")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.make_suite:
        if args.suite or args.model or args.out:
            parser.error("--make-suite is a separate mode from checkpoint evaluation")
        suite = rb.make_suite(args.corpus if args.corpus is not None else default_corpus_paths(),
                              seed=args.seed, groups_per_family=args.groups_per_family)
        rb.validate_suite(suite)
        write_new_json(args.make_suite, suite)
        print(json.dumps({"suite": str(args.make_suite.resolve()), "sha256": suite["sha256"],
                          "groups": len(suite["groups"]), "prompts": 6 * len(suite["groups"]),
                          "corpus_audit": suite["corpus_audit"]}, indent=2))
        return
    if not args.suite or not args.model or not args.out:
        parser.error("Evaluation requires --suite, at least one --model NAME=PATH, and --out")
    if args.corpus:
        parser.error("--corpus is only accepted when freezing a suite")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {args.out}")
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error("--threads and --max-new-tokens must be positive")
    suite_bytes = rb.file_sha256(args.suite)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    rb.validate_suite(suite)
    rb.verify_corpus(suite)
    groups = rb.select_groups(suite, args.limit)
    models = []
    for value in args.model:
        if "=" not in value:
            parser.error("Each --model must be NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in {n for n, _ in models}:
            parser.error("Model names must be nonempty and unique")
        models.append((name, Path(path)))
    code_paths = lineage.project_source_files(PROJECT)
    code_before = lineage.input_manifest(code_paths, PROJECT)
    import torch
    torch.set_num_threads(args.threads)
    decode = {"method": "native-greedy-kv-cache", "max_new_tokens": args.max_new_tokens,
              "threads": args.threads, "seed": args.seed, "prompt_truncation": False,
              "answer_budget": "min(requested, checkpoint context - prompt tokens)",
              "exhausted_budget_counts_as_failure": True, "all_grafts": True}
    report = {"schema": "supermix-semantic-reasoning-report-v1", "suite_sha256": suite["sha256"],
              "suite_file_sha256": suite_bytes, "suite_path": str(args.suite.resolve()),
              "selection": [g["id"] for g in groups], "selection_sha256": rb.canonical_sha256(groups),
              "scorer": rb.SCORER, "code_manifest": code_before, "decode": decode,
              "decode_sha256": rb.canonical_sha256(decode), "python": platform.python_version(),
              "torch": torch.__version__, "corpus_audit": suite["corpus_audit"],
              "promotion_allowed": False, "models": []}
    for name, path in models:
        report["models"].append(evaluate_one(name, path, groups, decode))
    rb.verify_corpus(suite)
    if rb.file_sha256(args.suite) != suite_bytes:
        raise ValueError("Suite file changed during evaluation")
    if lineage.input_manifest(lineage.project_source_files(PROJECT), PROJECT) != code_before:
        raise ValueError("Project source bytes changed during evaluation")
    report["sha256"] = rb.canonical_sha256(report)
    write_new_json(args.out, report)
    print(json.dumps({"report": str(args.out.resolve()), "models": {m["name"]: m["metrics"] for m in report["models"]}}, indent=2))


if __name__ == "__main__":
    main()
