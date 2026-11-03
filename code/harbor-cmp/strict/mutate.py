"""Mutation test of the graders: does a reference solution with one realistic mistake still pass?

  python harbor-cmp/strict/mutate.py [-j 16] [--tasks N]

For data-eng-bench tasks, copy the task with its reference solution's SQL changed by one mutation (first match
only): a shorter date window, an inner instead of a left join, a count without DISTINCT, or a strict instead of
an inclusive comparison. Each mutant runs as an oracle trial (validate_trial.sh) and is graded by the original test
(reward) and the strict check alone (strict). Mutants whose output equals the reference output (strict passes) are
equivalent: the mistake did not change the result. Writes strict/mutation_results.json.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parents[1]
OUT = H / "mutants"
MUTATIONS = {
    "date_window": (r"(\d+)(\s*(?:days?|DAYS?)\b|'\s*(?:day|DAY))", lambda m: f"{max(1, int(m.group(1)) // 2)}{m.group(2)}"),
    "join": (r"\bLEFT\s+JOIN\b", lambda m: "INNER JOIN"),
    "grain": (r"COUNT\(\s*DISTINCT\s+", lambda m: "COUNT("),
    "comparison": (r">=", lambda m: ">"),
}


def make(task, name):
    pat, repl = MUTATIONS[name]
    src = (H / f"tasks/{task}/solution/solve.sh").read_text()
    m = re.search(pat, src)
    if not m:
        return None
    dst = OUT / f"{task}__{name}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(H / f"tasks/{task}", dst, symlinks=True)
    (dst / "solution/solve.sh").write_text(src[:m.start()] + repl(m) + src[m.end():])
    return dst.name, m.group(0), repl(m)


def run(task, mutant):
    subprocess.run(["bash", str(H / "validate_trial.sh"), "oracle", mutant, "mutants"],
                   env={**os.environ, "TASKS": str(OUT), "MOUNT": task})
    found = list((H / f"jobs/validate-mutants/{mutant}").glob("*/result.json"))
    rewards = ((json.loads(found[0].read_text()).get("verifier_result") or {}).get("rewards") or {}) if found else {}
    return rewards.get("reward"), rewards.get("strict_check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=16)
    ap.add_argument("--tasks", type=int, default=0, help="limit to the first N validated data-eng-bench tasks")
    args = ap.parse_args()
    valid = json.load(open(H / "validated_tasks.json"))
    tasks = [t for t, v in sorted(valid.items()) if t.startswith("deb__") and v["ok"]]
    tasks = tasks[:args.tasks] if args.tasks else tasks
    jobs = [(t, made) for t in tasks for name in MUTATIONS if (made := make(t, name))]
    with ThreadPoolExecutor(args.j) as pool:
        res = list(pool.map(lambda j: run(j[0], j[1][0]), jobs))
    out = [{"task": t, "mutant": mid, "from": a, "to": b, "reward": r, "strict": s}
           for (t, (mid, a, b)), (r, s) in zip(jobs, res)]
    json.dump(out, open(H / "strict/mutation_results.json", "w"), indent=1)
    changed = [x for x in out if x["strict"] == 0 and x["reward"] is not None]
    caught_orig = sum(x["reward"] == 0 for x in changed)
    print(f"{len(out)} mutants; {len(changed)} changed the output (strict fails); original grader caught "
          f"{caught_orig}/{len(changed)}; equivalent (strict passes): {sum(x['strict'] == 1 for x in out)}")


if __name__ == "__main__":
    main()
