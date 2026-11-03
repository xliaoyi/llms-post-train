"""Run one harness-comparison config over the v3 test and train splits (4 attempts each) through Harbor.

  python harbor-cmp/driver.py --config alt-think -j 12

Each trial is harbor-cmp/run_trial.sh (fresh workspace, `harbor run`, Harbor's verifier). Only tasks that passed
preflight.py (validated_tasks.json) are run. Trials with a result.json are skipped; infrastructure failures
(environment/agent setup, model endpoint unreachable, no verifier result) are moved to jobs/<config>/_infra/ (kept,
not deleted) and rerun, up to --rounds times.
"""
import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harbor-cmp"
INFRA = ("Connection error", "ENOTFOUND", "ECONNREFUSED", "backend unreachable",  # fatal if in the log tail
         "streamIdleTimeout")


def result(cfg, split, task, attempt):
    """(reward, exception type, infra?) of a finished trial, or None if it has not run."""
    found = list((H / "jobs" / cfg / split / f"{task}-{attempt}").glob("*/result.json"))
    if not found:
        return None
    r = json.loads(found[0].read_text())
    exc = r.get("exception_info") or {}
    reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    log = "".join(p.read_text(errors="replace")[-3000:] for p in found[0].parent.glob("agent/*.txt"))
    infra = reward is None or any(s in exc.get("exception_message", "") + log for s in INFRA) \
        or "setup" in exc.get("exception_type", "").lower() or "Environment" in exc.get("exception_type", "")
    return reward or 0.0, exc.get("exception_type"), infra


def run(cfg, split, task, attempt):
    subprocess.run(["bash", str(H / "run_trial.sh"), cfg, task, str(attempt), split])
    return (split, task, attempt), result(cfg, split, task, attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["qwen-think", "qwen-nothink", "alt-think", "alt-nothink"])
    ap.add_argument("-j", type=int, default=12)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    validated = {t for t, v in json.load(open(H / "validated_tasks.json")).items() if v["ok"]}
    jobs = [(split, f"{t['source']}__{t['task']}", a) for split in ["test", "train"] for a in range(args.attempts)
            for t in json.load(open(ROOT / f"data/{split}_tasks_v3.json"))
            if f"{t['source']}__{t['task']}" in validated]
    print(f"{len(validated)} validated tasks, {len(jobs)} trials", flush=True)
    for rnd in range(1, args.rounds + 1):
        todo = []
        for j in jobs:
            r = result(args.config, *j)
            if r and r[2]:  # infrastructure failure: keep its record under _infra/ and rerun
                attic = H / "jobs" / args.config / "_infra" / j[0] / f"{j[1]}-{j[2]}.{int(time.time())}"
                attic.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(H / "jobs" / args.config / j[0] / f"{j[1]}-{j[2]}", attic)
            if not r or r[2]:
                todo.append(j)
        print(f"round {rnd}: {len(todo)} trials to run", flush=True)
        if not todo:
            break
        done = passed = 0
        with ThreadPoolExecutor(args.j) as pool:
            for fut in as_completed([pool.submit(run, args.config, *j) for j in todo]):
                (split, task, attempt), r = fut.result()
                done += 1
                passed += bool(r and r[0] == 1)
                print(f"[{done}/{len(todo)} pass={passed}] {split} {task}#{attempt} reward={r and r[0]} "
                      f"exc={r and r[1]} {'INFRA' if not r or r[2] else ''}", flush=True)
    print(f"{args.config} done", flush=True)


if __name__ == "__main__":
    main()
