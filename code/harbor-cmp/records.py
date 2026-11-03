"""One record per trial, then avg pass / SR@1 / SR@4 per config and split, on both the benchmark's own grader
(reward) and the strict semantic check (strict).

  python harbor-cmp/records.py qwen-think qwen-nothink alt-think alt-nothink

Writes jobs/<config>/records.jsonl and results.json. Each record keeps separately:
  outcome:      reward and strict from the verifier (None if the verifier gave none)
  termination:  why the agent stopped, from the harness's own log: completed | turn_limit | context_overflow |
                agent_timeout | api_error | crashed (exception type in `exception`)
  verifier:     ran | no_reward (verifier ran but wrote no reward) | not_run
plus the image and workspace hashes of the snapshot the trial started from (validated_tasks.json).
"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent


def termination(trial, exc):
    if exc == "AgentTimeoutError":
        return "agent_timeout"
    qwen = trial / "agent/qwen-code.txt"
    if qwen.exists():
        tail = qwen.read_text(errors="replace")[-5000:]
        if "Reached max session turns" in tail:
            return "turn_limit"
        if "API Error" in tail:
            return "api_error"
        return "completed" if exc is None else "crashed"
    term = None
    alt = trial / "agent/altimate.txt"
    for line in alt.read_text(errors="replace").splitlines() if alt.exists() else []:
        if '"type": "termination"' in line or '"type":"termination"' in line:
            term = json.loads(line)
    if term is None:
        return "crashed"
    if term.get("why_harness_stopped") == "budget-exhausted":
        return "turn_limit"
    if term.get("why_harness_stopped") == "error":
        return "context_overflow" if "Context budget exceeded" in alt.read_text(errors="replace") else "api_error"
    return "completed"


def record(cfg, split, task, attempt, validated):
    found = list((H / f"jobs/{cfg}/{split}/{task}-{attempt}").glob("*/result.json"))
    if not found:
        return None
    r = json.loads(found[0].read_text())
    exc = (r.get("exception_info") or {}).get("exception_type")
    vr = r.get("verifier_result")
    rewards = (vr or {}).get("rewards") or {}
    verifier = "ran" if "reward" in rewards else "no_reward" if (found[0].parent / "verifier").exists() else "not_run"
    snap = validated.get(task, {})
    return {"config": cfg, "split": split, "task": task, "attempt": attempt,
            "reward": rewards.get("reward"), "strict": rewards.get("strict"),
            "termination": termination(found[0].parent, exc), "exception": exc, "verifier": verifier,
            "image_sha256": snap.get("image_sha256"), "workspace_sha256": snap.get("workspace_sha256"),
            "trial_dir": str(found[0].parent.relative_to(H))}


def metrics(recs, key):
    by_task = {}
    for x in recs:
        by_task.setdefault(x["task"], {})[x["attempt"]] = (x[key] or 0) == 1
    n = len(by_task)
    return {"tasks": n, "avg": sum(sum(a.values()) for a in by_task.values()) / max(1, len(recs)),
            "sr1": sum(a.get(0, False) for a in by_task.values()) / max(1, n),
            "sr4": sum(any(a.values()) for a in by_task.values()) / max(1, n)}


def main():
    validated = json.load(open(H / "validated_tasks.json"))
    results = {}
    for cfg in sys.argv[1:]:
        recs = []
        for split in ["test", "train"]:
            for t in json.load(open(ROOT / f"data/{split}_tasks_v3.json")):
                task = f"{t['source']}__{t['task']}"
                if not validated.get(task, {}).get("ok"):
                    continue
                recs += [x for a in range(4) if (x := record(cfg, split, task, a, validated))]
        with open(H / f"jobs/{cfg}/records.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in recs)
        for split in ["test", "train"]:
            s = [x for x in recs if x["split"] == split]
            term = {}
            for x in s:
                term[x["termination"]] = term.get(x["termination"], 0) + 1
            results[f"{cfg}/{split}"] = {"trials": len(s), "reward": metrics(s, "reward"),
                                         "strict": metrics(s, "strict"), "termination": term}
            m, st = results[f"{cfg}/{split}"]["reward"], results[f"{cfg}/{split}"]["strict"]
            print(f"{cfg:13s} {split:5s} n={len(s):3d}  reward avg {m['avg']:.1%} SR@1 {m['sr1']:.1%} SR@4 "
                  f"{m['sr4']:.1%} | strict avg {st['avg']:.1%} SR@1 {st['sr1']:.1%} SR@4 {st['sr4']:.1%} | {term}")
    json.dump(results, open(H / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
