"""Sanity-check tasks: verifier must give 0 on the untouched state and 1 after the reference solution.

  python harness/oracle.py --source deb --category analytics -j 16
  python harness/oracle.py --source spider2 ade -j 16
"""
import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from env import ROOT, Env, Task, list_tasks


def check(source, task_id, out_root):
    out = out_root / source / task_id
    res = {"source": source, "task": task_id}
    t0 = time.time()
    try:
        task = Task(source, task_id)
        env = Env(task, out / "empty")
        res["empty"] = env.verify()
        env.close()
        env = Env(task, out / "oracle")
        env.oracle()
        res["oracle"] = env.verify()
        env.close()
    except Exception:
        res["error"] = traceback.format_exc()[-1500:]
    res["sec"] = round(time.time() - t0)
    res["ok"] = res.get("empty") == 0 and res.get("oracle") == 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", required=True, choices=["deb", "spider2", "ade"])
    ap.add_argument("--category")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "runs/oracle"))
    args = ap.parse_args()
    out_root = Path(args.out)
    jobs = [(s, t) for s in args.source for t in (args.tasks or list_tasks(s, args.category))]
    print(f"{len(jobs)} tasks", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.j) as pool, open(out_root / "results.jsonl", "a") as f:
        futs = [pool.submit(check, s, t, out_root) for s, t in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            status = "OK " if r["ok"] else "BAD"
            print(f"{status} {r['source']}/{r['task']} empty={r.get('empty')} oracle={r.get('oracle')} "
                  f"{r['sec']}s {'ERROR' if 'error' in r else ''}", flush=True)


if __name__ == "__main__":
    main()
