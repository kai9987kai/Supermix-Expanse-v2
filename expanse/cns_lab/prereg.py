"""Pre-registration for the Connectome Lab: hypothesis files, locks and immutable verdicts.

    python expanse/cns_lab/prereg.py lock H-CNS-001            # freeze the hypothesis before any run
    python expanse/cns_lab/prereg.py status                    # which runs each hypothesis still needs
    python expanse/cns_lab/prereg.py verdict H-CNS-001         # once, from the recorded run metrics only

A hypothesis (``hypotheses/<id>.json``) names a ``protocol`` shared by its arms
(base checkpoint, steps, batch, learning rates, data mix, eval points,
``base_config``), the ``arms`` (``LabConfig`` overrides), which arm is the
``control`` and which the ``treatment``, the ``seeds``, the ``primary_metric``
and its ``direction``, and the ``success_rule``::

    {"all_same_sign": true, "min_mean_delta_rel": 0.01}   # or "min_mean_delta": 0.05 (absolute)

``lock`` writes ``hypotheses/<id>.lock.json`` (sha256 of the file bytes, UTC
time, git commit). A lock is never replaced: editing a locked hypothesis makes
``run_lab`` and ``verdict`` refuse it. The verdict compares treatment and
control seed by seed at the protocol's final step, with ``delta`` signed so
that positive favours the treatment:

    SUPPORTED       every run finished, the rule holds
    NOT SUPPORTED   every run finished, the rule does not hold
    INCONCLUSIVE    a run is missing, failed, or its metric is not finite

and is written once to ``verdicts/<id>.verdict.json`` (never overwritten).

Runs are keyed by (protocol, full arm config, seed), so an arm shared by two
hypotheses with the same protocol is trained once.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT / "src"))

import connectome_lab as cl  # noqa: E402

HYP_DIR = LAB / "hypotheses"
OUT = ROOT / "checkpoints" / "cns_lab"
RUNS = OUT / "runs"
VERDICTS = LAB / "verdicts"
VERDICT_SCHEMA = "supermix-expanse-cns-lab-verdict-v1"
REQUIRED = ("id", "claim", "protocol", "arms", "control_arm", "treatment_arm", "seeds", "primary_metric", "direction",
            "success_rule")


class PreregError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hyp_path(hid: str, hyp_dir: Path = HYP_DIR) -> Path:
    return Path(hyp_dir) / f"{hid}.json"


def lock_path(hid: str, hyp_dir: Path = HYP_DIR) -> Path:
    return Path(hyp_dir) / f"{hid}.lock.json"


def validate(h: Dict[str, Any]) -> None:
    if h.get("exploratory"):  # no-claim arms: reported, never given a verdict
        missing = [k for k in ("id", "claim", "protocol", "arms", "seeds") if k not in h]
        if missing:
            raise PreregError(f"exploratory file missing {missing}")
        for over in h["arms"].values():
            cl.LabConfig.from_dict({**h["protocol"].get("base_config", {}), **over})
        return
    missing = [k for k in REQUIRED if k not in h]
    if missing:
        raise PreregError(f"hypothesis missing {missing}")
    for arm in (h["control_arm"], h["treatment_arm"]):
        if arm not in h["arms"]:
            raise PreregError(f"arm {arm!r} not defined")
    if h["direction"] not in ("lower_is_better", "higher_is_better"):
        raise PreregError("direction must be lower_is_better or higher_is_better")
    rule = h["success_rule"]
    if not ("min_mean_delta" in rule) ^ ("min_mean_delta_rel" in rule):
        raise PreregError("success_rule needs exactly one of min_mean_delta / min_mean_delta_rel")
    if len(h["seeds"]) < 2 or len(set(h["seeds"])) != len(h["seeds"]):
        raise PreregError("at least two distinct seeds")
    base = h["protocol"].get("base_config", {})
    for name, over in h["arms"].items():
        cl.LabConfig.from_dict({**base, **over})  # raises on a bad arm


def arm_config(h: Dict[str, Any], arm: str) -> "cl.LabConfig":
    return cl.LabConfig.from_dict({**h["protocol"].get("base_config", {}), **h["arms"][arm]})


def run_key(protocol: Dict[str, Any], cfg: "cl.LabConfig", seed: int) -> str:
    """Identity of one run. ``base_config`` is already folded into ``cfg``; notes do not count."""
    proto = {k: v for k, v in protocol.items() if k not in ("base_config", "notes")}
    blob = json.dumps({"protocol": proto, "config": cfg.to_dict(), "seed": int(seed)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def run_dir(h: Dict[str, Any], arm: str, seed: int, runs: Path = RUNS) -> Path:
    return Path(runs) / run_key(h["protocol"], arm_config(h, arm), seed)


def git_commit() -> Dict[str, Any]:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except Exception:  # pragma: no cover - no git
        return {"commit": None, "dirty": None}


def lock(hid: str, hyp_dir: Path = HYP_DIR) -> Dict[str, Any]:
    p = hyp_path(hid, hyp_dir)
    raw = p.read_bytes()
    h = json.loads(raw)
    if h.get("id") != hid:
        raise PreregError(f"{p} has id {h.get('id')!r}")
    validate(h)
    lp = lock_path(hid, hyp_dir)
    sha = sha256_bytes(raw)
    if lp.exists():
        old = json.loads(lp.read_text(encoding="utf-8"))
        if old["sha256"] != sha:
            raise PreregError(f"{hid} is already locked with a different file (locks are never replaced)")
        return old
    rec = {"id": hid, "sha256": sha, "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "git": git_commit(), "file": p.name}
    lp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def load_locked(hid: str, hyp_dir: Path = HYP_DIR) -> Dict[str, Any]:
    """The hypothesis, after checking it is locked and unchanged since."""
    p = hyp_path(hid, hyp_dir)
    lp = lock_path(hid, hyp_dir)
    if not lp.exists():
        raise PreregError(f"{hid} is not locked: run `prereg.py lock {hid}` before any run")
    raw = p.read_bytes()
    rec = json.loads(lp.read_text(encoding="utf-8"))
    if sha256_bytes(raw) != rec["sha256"]:
        raise PreregError(f"{hid} changed after it was locked (sha mismatch): refusing")
    h = json.loads(raw)
    validate(h)
    h["_lock"] = rec
    return h


def read_metric(metrics: Dict[str, Any], name: str, step: Optional[int] = None) -> Optional[float]:
    """``name`` at the final eval (or at ``step``) of a metrics.json dict; None if absent."""
    evals = metrics.get("evals") or []
    if not evals:
        return None
    ev = evals[-1] if step is None else next((e for e in evals if int(e["step"]) == int(step)), None)
    if ev is None:
        return None
    v = ev.get(name)
    return float(v) if isinstance(v, (int, float)) else None


def evaluate_rule(h: Dict[str, Any], runs: Path = RUNS) -> Dict[str, Any]:
    """The verdict from the recorded metrics only."""
    final_step = int(h["protocol"]["steps"])
    sign = 1.0 if h["direction"] == "lower_is_better" else -1.0
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for seed in h["seeds"]:
        row: Dict[str, Any] = {"seed": int(seed)}
        for role in ("control", "treatment"):
            arm = h[f"{role}_arm"]
            d = run_dir(h, arm, seed, runs)
            mp = d / "metrics.json"
            val = None
            if not mp.exists():
                problems.append(f"{arm} seed {seed}: no run")
            else:
                m = json.loads(mp.read_text(encoding="utf-8"))
                if m.get("status") != "done":
                    problems.append(f"{arm} seed {seed}: status {m.get('status')}")
                elif int(m.get("steps_done", -1)) != final_step and not m.get("untrainable"):
                    problems.append(f"{arm} seed {seed}: {m.get('steps_done')} steps, protocol {final_step}")
                else:
                    val = read_metric(m, h["primary_metric"])
                    if val is None or not math.isfinite(val):
                        problems.append(f"{arm} seed {seed}: metric {h['primary_metric']} = {val}")
                        val = None
            row[role] = val
            row[f"{role}_run"] = d.name
        if row["control"] is not None and row["treatment"] is not None:
            row["delta"] = sign * (row["control"] - row["treatment"])  # > 0 favours the treatment
        rows.append(row)
    out: Dict[str, Any] = {"schema": VERDICT_SCHEMA, "id": h["id"], "claim": h["claim"],
                           "primary_metric": h["primary_metric"], "direction": h["direction"],
                           "success_rule": h["success_rule"], "per_seed": rows, "problems": problems,
                           "hypothesis_sha256": (h.get("_lock") or {}).get("sha256")}
    if problems:
        out["verdict"] = "INCONCLUSIVE"
        return out
    deltas = [r["delta"] for r in rows]
    mean_delta = sum(deltas) / len(deltas)
    rule = h["success_rule"]
    if "min_mean_delta" in rule:
        threshold = float(rule["min_mean_delta"])
    else:
        control_mean = sum(r["control"] for r in rows) / len(rows)
        threshold = float(rule["min_mean_delta_rel"]) * abs(control_mean)
    same_sign = all(d > 0 for d in deltas)
    ok = (same_sign or not rule.get("all_same_sign", True)) and mean_delta >= threshold
    out.update({"mean_delta": mean_delta, "threshold": threshold, "all_same_sign": same_sign,
                "n_favour_treatment": sum(d > 0 for d in deltas), "n_seeds": len(deltas),
                "verdict": "SUPPORTED" if ok else "NOT SUPPORTED"})
    return out


def verdict(hid: str, hyp_dir: Path = HYP_DIR, runs: Path = RUNS, verdict_dir: Path = VERDICTS,
            write: bool = True) -> Dict[str, Any]:
    vp = Path(verdict_dir) / f"{hid}.verdict.json"
    h = load_locked(hid, hyp_dir)
    if h.get("exploratory"):
        raise PreregError(f"{hid} is exploratory: no verdict (see analyze.py)")
    if vp.exists():
        old = json.loads(vp.read_text(encoding="utf-8"))
        if old.get("hypothesis_sha256") != h["_lock"]["sha256"]:
            raise PreregError(f"{vp} was issued for a different hypothesis file")
        old["_note"] = "existing verdict (verdicts are immutable)"
        return old
    res = evaluate_rule(h, runs)
    res["issued_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res["git"] = git_commit()
    if write and res["verdict"] != "INCONCLUSIVE":
        vp.parent.mkdir(parents=True, exist_ok=True)
        vp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    elif write:
        res["_note"] = "INCONCLUSIVE verdicts are reported, not written: finish the runs"
    return res


def all_hypotheses(hyp_dir: Path = HYP_DIR) -> List[str]:
    return sorted(p.stem for p in Path(hyp_dir).glob("*.json") if not p.name.endswith(".lock.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock").add_argument("ids", nargs="+")
    sub.add_parser("verdict").add_argument("ids", nargs="+")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    if args.cmd == "lock":
        for hid in args.ids:
            print(json.dumps(lock(hid), indent=1))
        return 0
    if args.cmd == "verdict":
        for hid in args.ids:
            print(json.dumps(verdict(hid), indent=1))
        return 0
    for hid in all_hypotheses():
        try:
            h = load_locked(hid)
        except PreregError as exc:
            print(f"{hid}: {exc}")
            continue
        arms = list(h["arms"]) if h.get("exploratory") else [h["control_arm"], h["treatment_arm"]]
        for arm in arms:
            for s in h["seeds"]:
                d = run_dir(h, arm, s)
                m = d / "metrics.json"
                st = json.loads(m.read_text(encoding="utf-8")).get("status") if m.exists() else "todo"
                print(f"{hid} {arm:>14} seed {s}: {st} ({d.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
