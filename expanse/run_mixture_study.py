"""Run a byte-bound, matched continuation pilot without changing the chat model.

Prepare after freezing a reasoning suite; run natural and balanced arms serially.
The manifest is immutable. Completed arms are reused only after hash validation.
This pilot does not select or promote a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from cns_lab.lineage import canonical_sha256, file_sha256, input_manifest, project_source_files

SCHEMA = "supermix-mixture-study-v1"
ARMS = ("natural", "balanced")


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bound_inputs(parent: Path, suite: Path) -> list[Path]:
    data = ROOT / "data"
    corpus = PROJECT / "supermix-archimedes" / "corpus"
    return [parent, suite, *project_source_files(PROJECT),
            *(corpus / f"{name}.jsonl" for name in ("omni", "code", "math")),
            data / "code_rows.jsonl", data / "bio_rows.clean.jsonl",
            data / "v3" / "connectome_rows_v3.jsonl",
            *(data / "v3" / f"fresh_{name}.jsonl" for name in ("omni", "code", "math"))]


def verify_manifest(manifest: dict) -> None:
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("schema") != SCHEMA or canonical_sha256(body) != manifest.get("sha256"):
        raise ValueError("Invalid or edited study manifest")
    current = input_manifest(bound_inputs(Path(manifest["parent"]), Path(manifest["suite"])), PROJECT)
    if current != manifest["inputs"]:
        raise ValueError("Study input bytes/source inventory changed; preserve this run and prepare a new directory")
    if file_sha256(Path(manifest["initial_cache"])) != manifest["initial_cache_sha256"]:
        raise ValueError("Initial feature cache changed")


def prepare(args) -> None:
    output = args.directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Prepare requires an empty directory; historical studies are never overwritten")
    parent, suite = args.parent.resolve(), args.suite.resolve()
    from src.reasoning_benchmark import load_suite
    load_suite(suite)
    inputs = input_manifest(bound_inputs(parent, suite), PROJECT)
    initial_cache = output / "omni7.initial.pt"
    shutil.copyfile(ROOT / "data" / "omni7_states.pt", initial_cache)
    common = ["--inp", str(parent), "--steps", str(args.steps), "--batch", "4",
              "--threads", str(args.threads), "--seed", "2026", "--sampling_seed", "2026",
              "--v3_data", "on", "--fly_rows", "1500", "--max_rows_per_source", "12000",
              "--dev_cap", "24", "--lr_trunk", "1e-5", "--lr_graft", "1e-4",
              "--kd_arch", "0", "--wake", "preserve", "--eval_every", str(args.steps),
              "--save_every", "20", "--resume", "auto"]
    commands = {}
    for arm in ARMS:
        commands[arm] = [sys.executable, "-u", str(ROOT / "train_expanse.py"), *common,
                         "--sampling", arm, "--out", str(output / f"{arm}.pt"),
                         "--omni7_cache", str(output / f"omni7.{arm}.pt")]
    body = {"schema": SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent": str(parent), "parent_sha256": file_sha256(parent), "suite": str(suite), "inputs": inputs,
            "initial_cache": str(initial_cache), "initial_cache_sha256": file_sha256(initial_cache),
            "commands": commands, "python": sys.version, "platform": platform.platform(),
            "design": {"steps_per_arm": args.steps, "batch": 4, "paired_seed": 2026,
                       "changed_variable": "source row sampling probabilities and resulting row order",
                       "balanced_weights": "equal probability for every nonempty packed source",
                       "primary": "frozen generated-answer accuracy and all-correct semantic groups",
                       "secondary": "reply-token dev loss by source; actual row/token exposure",
                       "promotion": False,
                       "limits": "one short paired seed; equal updates, not equal tokens or FLOPs; no general capability claim"}}
    write_new(output / "manifest.json", {**body, "sha256": canonical_sha256(body)})
    print(f"Prepared {output / 'manifest.json'}", flush=True)


def validate_training(tr: dict, arm: str, manifest: dict) -> None:
    command = manifest["commands"][arm][3:]
    options = dict(zip(command[::2], command[1::2]))
    for key, cast in (("steps", int), ("batch", int), ("seed", int), ("lr_trunk", float),
                      ("lr_graft", float), ("kd_arch", float), ("wake", str)):
        if tr.get(key) != cast(options[f"--{key}"]) or (cast is int and type(tr[key]) is not int):
            raise ValueError(f"{arm} receipt differs from command: {key}")
    if (tr.get("target_steps") != tr["steps"] or tr.get("ended_by") != "done"
            or tr.get("smoke") is not False or tr.get("parent_sha256") != manifest["parent_sha256"]):
        raise ValueError(f"{arm} receipt has wrong parent or incomplete training")
    corpus = tr.get("corpus_args", {})
    for key in ("fly_rows", "max_rows_per_source", "dev_cap", "seed"):
        if corpus.get(key) != int(options[f"--{key}"]):
            raise ValueError(f"{arm} corpus setting differs: {key}")
    if corpus.get("v3_data") is not True:
        raise ValueError(f"{arm} did not use v3 data")
    sampling = tr.get("sampling", {})
    if sampling.get("mode") != arm or sampling.get("seed") != int(options["--sampling_seed"]):
        raise ValueError(f"{arm} sampling mode/seed differs")
    pools, probabilities = sampling.get("pool_rows", {}), sampling.get("source_row_probabilities", {})
    if not pools or set(pools) != set(probabilities) or any(type(n) is not int or n <= 0 for n in pools.values()):
        raise ValueError(f"{arm} invalid packed source pools")
    for source, count in pools.items():
        expected = 1 / len(pools) if arm == "balanced" else count / sum(pools.values())
        if not math.isclose(probabilities[source], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{arm} source probabilities differ")
    exposure = sampling.get("exposure", {})
    if (exposure.get("complete") is not True or exposure.get("since_step") != 0
            or exposure.get("updates") != tr["steps"]
            or sum(exposure.get("rows", {}).values()) != tr["steps"] * tr["batch"]):
        raise ValueError(f"{arm} incomplete exposure receipt")
    for key in ("rows", "supervised_tokens"):
        values = exposure.get(key, {})
        if set(values) != set(pools) or any(type(n) is not int or n < 0 for n in values.values()):
            raise ValueError(f"{arm} invalid exposure {key}")
    if any(type(exposure.get(k)) is not int for k in ("since_step", "updates")):
        raise ValueError(f"{arm} invalid exposure step types")
    contract = tr.get("resume_contract", {})
    for key in ("parent_sha256", "data_fingerprint"):
        if not tr.get(key) or contract.get(key) != tr[key]:
            raise ValueError(f"{arm} resume contract differs: {key}")
    config = {k: v for k, v in sampling.items() if k not in ("weight_unit", "exposure")}
    if contract.get("sampling") != config:
        raise ValueError(f"{arm} resume contract sampling differs")
    arguments = contract.get("arguments", {})
    for key, value in arguments.items():
        if f"--{key}" in options and str(value) != options[f"--{key}"]:
            # Numeric command strings may use scientific notation.
            try:
                same = float(value) == float(options[f"--{key}"])
            except (ValueError, TypeError):
                same = False
            if not same:
                raise ValueError(f"{arm} resume contract argument differs: {key}")
        if key in tr and tr[key] != value and not (key == "v2_balance" and tr[key] is None):
            raise ValueError(f"{arm} inconsistent training argument: {key}")
    if any(key not in arguments for key in ("threads", "eval_every", "steps", "wake", "seed")):
        raise ValueError(f"{arm} missing resume contract arguments")


def verify_completed(directory: Path, arm: str, manifest: dict) -> dict:
    run = json.loads((directory / f"{arm}.run.json").read_text(encoding="utf-8"))
    if (run.get("status") != "complete" or run.get("manifest_sha256") != manifest["sha256"]
            or run.get("command") != manifest["commands"][arm]):
        raise ValueError(f"Unverifiable completed {arm} arm")
    for kind, suffix in (("checkpoint", ".pt"), ("receipt", ".receipt.json")):
        if file_sha256(directory / f"{arm}{suffix}") != run[f"{kind}_sha256"]:
            raise ValueError(f"Changed {arm} {kind}")
    if file_sha256(directory / f"omni7.{arm}.pt") != run["feature_cache_sha256"]:
        raise ValueError(f"Changed {arm} feature cache")
    receipt = json.loads((directory / f"{arm}.receipt.json").read_text(encoding="utf-8"))
    tr = receipt["training"]
    if receipt["stage"] != "trained" or tr["steps"] != manifest["design"]["steps_per_arm"]:
        raise ValueError(f"Incomplete {arm} training")
    validate_training(tr, arm, manifest)
    return tr


def train(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        verify_manifest(manifest)
        if (directory / f"{arm}.run.json").exists():
            verify_completed(directory, arm, manifest)
            print(f"Verified completed {arm}; reusing", flush=True)
            continue
        if (directory / f"{arm}.pt").exists():
            raise ValueError(f"Unreceipted {arm} checkpoint exists; inspect it instead of overwriting")
        cache = directory / f"omni7.{arm}.pt"
        # Always reconstruct this study's regenerable arm cache from the bound
        # initial bytes, including on crash resume. Missing entries are encoded
        # deterministically by the same frozen parent encoder.
        shutil.copyfile(manifest["initial_cache"], cache)
        start = time.time()
        command = manifest["commands"][arm]
        with (directory / f"{arm}.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(f"[{arm}] {line}", end="", flush=True)
                log.write(line)
                log.flush()
            returncode = proc.wait()
        if returncode:
            raise RuntimeError(f"{arm} trainer exited {returncode}; see {arm}.log")
        verify_manifest(manifest)
        checkpoint, receipt = directory / f"{arm}.pt", directory / f"{arm}.receipt.json"
        if not checkpoint.exists() or not receipt.exists():
            raise RuntimeError(f"{arm} stopped before completion; rerun to resume its exact partial")
        write_new(directory / f"{arm}.run.json", {
            "status": "complete", "manifest_sha256": manifest["sha256"], "command": command,
            "checkpoint_sha256": file_sha256(checkpoint), "receipt_sha256": file_sha256(receipt),
            "feature_cache_sha256": file_sha256(cache), "elapsed_seconds": round(time.time() - start, 2)})
        verify_completed(directory, arm, manifest)
    left, right = [verify_completed(directory, arm, manifest) for arm in ARMS]
    for key in ("data_fingerprint", "parent_sha256", "dev_keys", "packed", "corpus_args"):
        if not left.get(key) or left[key] != right.get(key):
            raise ValueError(f"Matched-arm invariant failed: {key}")
    print("Both matched arms completed; run frozen generation evaluation. No model was promoted.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "train"))
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--parent", type=Path, default=ROOT / "checkpoints" / "supermix_expanse_v4.pt")
    ap.add_argument("--suite", type=Path)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.steps <= 0 or args.threads <= 0:
        ap.error("steps and threads must be positive")
    if args.stage == "prepare":
        if args.suite is None:
            ap.error("prepare requires --suite")
        prepare(args)
    else:
        train(args.directory.resolve())


if __name__ == "__main__":
    main()
