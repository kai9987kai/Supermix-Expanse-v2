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

``all_same_sign`` is a historical name (it is in locked files, so it stays):
it means **every seed favours the treatment** (every ``delta > 0``), not
"every delta has the same sign" -- three negative deltas do not meet it.

``lock`` writes ``hypotheses/<id>.lock.json`` (sha256 of the file bytes, UTC
time, git commit). A lock is never replaced: editing a locked hypothesis makes
``run_lab`` and ``verdict`` refuse it. The verdict compares treatment and
control seed by seed at the protocol's final step, with ``delta`` signed so
that positive favours the treatment:

    SUPPORTED       every run finished, the rule holds
    NOT SUPPORTED   every run finished, the rule does not hold
    INCONCLUSIVE    a run is missing, failed, or its metric is not finite, or a
                    recorded run is not provably the run the hypothesis asks for

and is written once to ``verdicts/<id>.verdict.json`` (never overwritten).

Before a run's metric is used, ``evaluate_rule`` checks that the recorded run
is the one the hypothesis asks for: ``run_key`` equals its directory name, the
recorded arm / seed / full ``LabConfig`` match, it was not an ``--unlocked``
run, its recorded lock hash is the hypothesis' lock (for a run shared with
another hypothesis: that hypothesis' lock, and its arm has the same config),
and a trainable run's last eval is at the protocol's final step. Any failure is
a problem (-> INCONCLUSIVE). Verdict schema version 2 (``verdict_schema_version``)
adds ``all_favour_treatment`` (same value as ``all_same_sign``) and ``inputs``
(sha256 of every metrics.json read). Verdict files issued before version 2
are never rewritten.

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
#: Version of the verdict *content* (2: run provenance checks, ``inputs``, ``all_favour_treatment``).
#: ``schema`` keeps its v1 string so readers of issued verdicts keep working.
VERDICT_SCHEMA_VERSION = 2
REQUIRED = ("id", "claim", "protocol", "arms", "control_arm", "treatment_arm", "seeds", "primary_metric", "direction",
            "success_rule")


class PreregError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    """sha256 of file bytes with line endings normalised (a CRLF checkout must not break a lock)."""
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


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


def _lock_sha(hid: str, hyp_dir: Path) -> Optional[str]:
    """sha256 recorded in ``<hid>.lock.json`` (None when there is no readable lock)."""
    lp = lock_path(hid, hyp_dir)
    try:
        return json.loads(lp.read_text(encoding="utf-8")).get("sha256")
    except (OSError, ValueError, AttributeError):
        return None


def check_run(h: Dict[str, Any], arm: str, seed: int, rdir: Path, m: Dict[str, Any],
              hyp_dir: Path = HYP_DIR) -> List[str]:
    """Why the recorded run ``m`` (read from ``rdir``) is not provably the run ``(h, arm, seed)``; [] if it is.

    A run shared with another hypothesis (``m['hypothesis']`` names that one: same
    protocol, config and seed, so the same run key) is accepted when that
    hypothesis defines the recorded arm with the same config and the recorded
    lock hash is that hypothesis' lock.
    """
    probs: List[str] = []
    if m.get("run_key") != rdir.name:
        probs.append(f"run_key {m.get('run_key')!r} != directory {rdir.name}")
    try:
        seed_ok = int(m.get("seed")) == int(seed)
    except (TypeError, ValueError):
        seed_ok = False
    if not seed_ok:
        probs.append(f"recorded seed {m.get('seed')!r} != {seed}")
    want_cfg = arm_config(h, arm).to_dict()
    if m.get("config") != want_cfg:
        probs.append("recorded config differs from the arm's LabConfig")
    if m.get("unlocked"):
        probs.append("run made from an unlocked hypothesis (--unlocked)")
    owner = m.get("hypothesis")
    lock_rec = m.get("lock") if isinstance(m.get("lock"), dict) else None
    if owner is None or owner == h["id"]:
        if m.get("arm") != arm:
            probs.append(f"recorded arm {m.get('arm')!r} != {arm!r}")
        want_sha = (h.get("_lock") or {}).get("sha256")  # None: an unlocked dict, nothing to compare
    else:  # trained once for another hypothesis with the same protocol
        try:
            other = json.loads(hyp_path(owner, hyp_dir).read_text(encoding="utf-8"))
            other_cfg = arm_config(other, m.get("arm")).to_dict() if m.get("arm") in (other.get("arms") or {}) else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            other, other_cfg = None, None
        if other is None:
            probs.append(f"recorded for {owner}, whose hypothesis file is missing or unreadable")
        elif other_cfg != want_cfg:
            probs.append(f"recorded as {owner}/{m.get('arm')}, which is not this arm's config")
        want_sha = _lock_sha(owner, hyp_dir)
        if lock_rec and want_sha is None:  # a shared run must trace back to a readable lock
            probs.append(f"recorded for {owner}, which has no readable lock")
    if lock_rec and lock_rec.get("sha256") is not None and want_sha is not None:
        if lock_rec["sha256"] != want_sha:
            probs.append(f"recorded lock sha {str(lock_rec['sha256'])[:12]} != hypothesis lock {want_sha[:12]}")
        if lock_rec.get("id") is not None and lock_rec["id"] != (owner or h["id"]):
            probs.append(f"recorded lock is for {lock_rec['id']!r}, run recorded for {owner or h['id']!r}")
    if not m.get("untrainable"):
        evals = m.get("evals") or []
        last = evals[-1].get("step") if evals and isinstance(evals[-1], dict) else None
        try:
            last_ok = last is not None and int(last) == int(h["protocol"]["steps"])
        except (TypeError, ValueError):
            last_ok = False
        if not last_ok:
            probs.append(f"last eval at step {last}, protocol {h['protocol']['steps']}")
    return probs


def evaluate_rule(h: Dict[str, Any], runs: Path = RUNS, hyp_dir: Path = HYP_DIR) -> Dict[str, Any]:
    """The verdict from the recorded metrics only (every run's provenance checked, see :func:`check_run`)."""
    final_step = int(h["protocol"]["steps"])
    sign = 1.0 if h["direction"] == "lower_is_better" else -1.0
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    inputs: List[Dict[str, Any]] = []
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
                raw = mp.read_bytes()
                inputs.append({"role": role, "arm": arm, "seed": int(seed), "run": d.name,
                               "metrics_sha256": hashlib.sha256(raw).hexdigest()})
                try:
                    m = json.loads(raw.decode("utf-8"))
                    if not isinstance(m, dict):
                        raise ValueError("not a JSON object")
                except ValueError as exc:
                    m = None
                    problems.append(f"{arm} seed {seed}: unreadable metrics.json ({exc})")
                if m is None:
                    pass
                elif m.get("status") != "done":
                    problems.append(f"{arm} seed {seed}: status {m.get('status')}")
                elif int(m.get("steps_done", -1)) != final_step and not m.get("untrainable"):
                    problems.append(f"{arm} seed {seed}: {m.get('steps_done')} steps, protocol {final_step}")
                else:
                    bad = check_run(h, arm, seed, d, m, hyp_dir)
                    if bad:
                        problems += [f"{arm} seed {seed}: {p}" for p in bad]
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
    out: Dict[str, Any] = {"schema": VERDICT_SCHEMA, "verdict_schema_version": VERDICT_SCHEMA_VERSION,
                           "id": h["id"], "claim": h["claim"],
                           "primary_metric": h["primary_metric"], "direction": h["direction"],
                           "success_rule": h["success_rule"], "per_seed": rows, "problems": problems,
                           "hypothesis_sha256": (h.get("_lock") or {}).get("sha256"), "inputs": inputs}
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
    # the rule key "all_same_sign" (locked files) means: every seed favours the treatment (delta > 0)
    all_favour = all(d > 0 for d in deltas)
    ok = (all_favour or not rule.get("all_same_sign", True)) and mean_delta >= threshold
    out.update({"mean_delta": mean_delta, "threshold": threshold, "all_same_sign": all_favour,
                "all_favour_treatment": all_favour,
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
    res = evaluate_rule(h, runs, hyp_dir)
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
