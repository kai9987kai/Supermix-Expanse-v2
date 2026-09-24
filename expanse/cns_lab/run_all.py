"""Run every pre-registered (hypothesis, arm, seed) once, then issue the verdicts.

    python expanse/cns_lab/run_all.py --threads 8                     # all locked hypotheses
    python expanse/cns_lab/run_all.py --hypotheses H-CNS-001 --exploratory

Each run is a separate ``run_lab.py`` process (memory is returned between
runs). Runs are deduplicated by run key, so an arm shared by several
hypotheses trains once, and ordered seed-major (every arm of seed 1, then seed
2, ...) so an interrupted sweep still leaves complete pairs. Finished runs are
skipped, so rerunning the command resumes. Afterwards: ``prereg.verdict`` for
every hypothesis whose runs all finished, and ``analyze.py`` over the runs.
Output is appended to ``external/logs/cns_lab.log``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(ROOT / "src"))

import prereg  # noqa: E402

LOG = ROOT.parent / "external" / "logs" / "cns_lab.log"


def say(msg: str) -> None:
    line = f"[run_all {dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def plan(ids, include_exploratory: bool):
    jobs, seen = [], set()
    hyps = []
    for hid in ids:
        h = prereg.load_locked(hid)
        if h.get("exploratory") and not include_exploratory:
            continue
        hyps.append(h)
    seeds = sorted({int(s) for h in hyps for s in h["seeds"]})
    for s in seeds:
        for h in hyps:
            if s not in h["seeds"]:
                continue
            arms = list(h["arms"]) if h.get("exploratory") else [h["treatment_arm"], h["control_arm"]]
            for arm in arms:
                key = prereg.run_dir(h, arm, s).name
                if key in seen:
                    continue
                seen.add(key)
                jobs.append((h["id"], arm, s, key))
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hypotheses", default="", help="comma list (default: every locked hypothesis)")
    ap.add_argument("--exploratory", action="store_true", help="also run exploratory (no-claim) arms")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="print the plan only")
    args = ap.parse_args(argv)
    ids = [x for x in args.hypotheses.split(",") if x] or prereg.all_hypotheses()
    jobs = plan(ids, args.exploratory)
    say(f"plan: {len(jobs)} runs for {ids} (exploratory {'on' if args.exploratory else 'off'})")
    for hid, arm, s, key in jobs:
        m = prereg.RUNS / key / "metrics.json"
        st = json.loads(m.read_text(encoding="utf-8")).get("status") if m.exists() else "todo"
        say(f"  {hid:>12} {arm:>18} seed {s} [{key}] {st}")
    if args.dry:
        return 0
    failed = []
    for i, (hid, arm, s, key) in enumerate(jobs, 1):
        m = prereg.RUNS / key / "metrics.json"
        if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("status") == "done":
            continue
        say(f"run {i}/{len(jobs)}: {hid} {arm} seed {s}")
        t0 = time.time()
        with open(LOG, "a", encoding="utf-8") as f:
            rc = subprocess.run([sys.executable, str(LAB / "run_lab.py"), "--hypothesis", hid, "--arm", arm,
                                 "--seed", str(s), "--threads", str(args.threads)],
                                stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
        say(f"run {i}/{len(jobs)} exit {rc} after {(time.time() - t0) / 60:.1f} min")
        if rc != 0:
            failed.append((hid, arm, s))
    for hid in ids:
        try:
            h = prereg.load_locked(hid)
            if h.get("exploratory"):
                continue
            v = prereg.verdict(hid)
            say(f"verdict {hid}: {v['verdict']}" + (f" (mean delta {v['mean_delta']:.5f}, threshold "
                                                     f"{v['threshold']:.5f}, {v['n_favour_treatment']}/{v['n_seeds']} "
                                                     f"seeds favour treatment)" if "mean_delta" in v else
                                                     f" problems: {v.get('problems')}"))
        except prereg.PreregError as exc:
            say(f"verdict {hid}: {exc}")
    subprocess.run([sys.executable, str(LAB / "analyze.py")], cwd=str(ROOT))
    say(f"finished; failed runs: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
