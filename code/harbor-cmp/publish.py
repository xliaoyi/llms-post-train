"""Publish a harness-comparison run: one Harbor job per config and split (for `harbor upload`), and Hugging Face
subsets in the dbt-agent-sft layout (messages, tools, source, metadata). For a run with records.py output
(after the environment fixes), metadata also has the strict reward and records.py's termination.

  .venv-harbor25/bin/python harbor-cmp/publish.py hub --jobs harbor-cmp/jobs-v1 --out harbor-cmp/hub-v1 --tag v1
  .venv/bin/python harbor-cmp/publish.py hf --jobs harbor-cmp/jobs-v1 --out hf/dbt-agent-sft --tag v1

driver.py runs every trial as its own one-trial Harbor job (<config>/<split>/<task>-<attempt>/<trial>/); `hub` copies
the trials of a config and split into one job dir with a merged config, lock and result (Harbor's own models).
altimate-code trials have no ATIF trajectory (only its JSON event log, agent/altimate.txt); `hub` adds
agent/trajectory.json built from it. Twilio account SIDs in trajectories are redacted, as on GitHub.
"""
import argparse
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

H = Path(__file__).resolve().parent
CONFIGS = {"qwen-think": ("qwen-code", True), "qwen-nothink": ("qwen-code", False),
           "alt-think": ("altimate-code", True), "alt-nothink": ("altimate-code", False)}
SID = re.compile(r"AC[0-9a-f]{32}")
MODEL = "Qwen/Qwen3.6-35B-A3B"


def trials(jobs, cfg, split):
    """(task, attempt, trial dir) of every trial with a result."""
    out = []
    for job in sorted((jobs / cfg / split).iterdir()):
        if not job.is_dir():
            continue
        task, attempt = job.name.rsplit("-", 1)
        for t in job.iterdir():
            if (t / "result.json").exists():
                out.append((task, int(attempt), t))
    return out


def altimate_events(trial):
    out = []
    f = trial / "agent/altimate.txt"
    for line in f.read_text(errors="replace").splitlines() if f.exists() else []:
        if line.startswith("{"):  # event lines; the rest is the CLI's console output
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def altimate_steps(events):
    """Group altimate-code events by model message: [(text, [tool parts], step_finish part)]."""
    msgs = {}
    for e in events:
        p = e.get("part") or {}
        if "messageID" not in p:
            continue
        m = msgs.setdefault(p["messageID"], {"text": [], "tools": [], "finish": None, "ts": e["timestamp"]})
        if e["type"] == "text":
            m["text"].append(p.get("text") or "")
        elif e["type"] == "tool_use":
            m["tools"].append(p)
        elif e["type"] == "step_finish":
            m["finish"] = p
    return list(msgs.values())


def instruction(task):
    return (H / f"tasks/{task}/instruction.md").read_text()


def to_messages(cfg, task, trial):
    """OpenAI-style chat messages of a trial (the user instruction, then assistant turns and tool results)."""
    msgs = []
    if CONFIGS[cfg][0] == "qwen-code":
        d = json.loads(SID.sub("AC_REDACTED_SID", (trial / "agent/trajectory.json").read_text()))
        for s in d["steps"]:
            if s["source"] != "agent":
                msgs.append({"role": s["source"], "content": s.get("message") or ""})
                continue
            msgs.append({"role": "assistant", "content": s.get("message") or "",
                         "reasoning_content": s.get("reasoning_content"),
                         "tool_calls": [{"id": c["tool_call_id"], "type": "function",
                                         "function": {"name": c["function_name"], "arguments": json.dumps(c["arguments"])}}
                                        for c in s.get("tool_calls") or []] or None})
            for r in (s.get("observation") or {}).get("results") or []:
                msgs.append({"role": "tool", "content": r.get("content") or "", "tool_call_id": r.get("source_call_id")})
        return msgs
    msgs.append({"role": "user", "content": instruction(task)})
    for m in altimate_steps(altimate_events(trial)):
        msgs.append({"role": "assistant", "content": SID.sub("AC_REDACTED_SID", "\n".join(m["text"])),
                     "reasoning_content": None,
                     "tool_calls": [{"id": p["callID"], "type": "function",
                                     "function": {"name": p["tool"], "arguments": json.dumps(p["state"].get("input") or {})}}
                                    for p in m["tools"]] or None})
        for p in m["tools"]:
            st = p["state"]
            out = st.get("output") if st.get("status") == "completed" else f"[{st.get('status')}] {st.get('error', '')}"
            msgs.append({"role": "tool", "content": SID.sub("AC_REDACTED_SID", out or ""), "tool_call_id": p["callID"]})
    return msgs


def termination(cfg, trial):
    if CONFIGS[cfg][0] == "qwen-code":
        f = trial / "agent/qwen-code.txt"
        txt = f.read_text(errors="replace") if f.exists() else ""
        return "turn_limit" if "Reached max session turns" in txt else "api_error" if "API Error" in txt else "completed"
    t = [e for e in altimate_events(trial) if e["type"] == "termination"]
    return (t[-1].get("why_harness_stopped") if t and t[-1].get("why_harness_stopped") != "none"
            else t[-1].get("done_reason") if t else "no_termination_event")


def hub(args):
    from harbor.models.job.config import JobConfig
    from harbor.models.job.lock import JobLock
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trajectories import Agent, Observation, ObservationResult, Step, ToolCall, Trajectory
    from harbor.models.trajectories.final_metrics import FinalMetrics
    from harbor.models.trial.result import TrialResult

    def alt_atif(task, trial):
        events = altimate_events(trial)
        steps = [Step(step_id=1, source="user", message=instruction(task))]
        tin = tout = 0
        for m in altimate_steps(events):
            tok = (m["finish"] or {}).get("tokens") or {}
            tin += tok.get("input") or 0
            tout += tok.get("output") or 0
            calls = [ToolCall(tool_call_id=p["callID"], function_name=p["tool"], arguments=p["state"].get("input") or {})
                     for p in m["tools"]]
            res = [ObservationResult(source_call_id=p["callID"], content=SID.sub("AC_REDACTED_SID", p["state"].get("output")
                                     or f"[{p['state'].get('status')}] {p['state'].get('error', '')}")) for p in m["tools"]]
            steps.append(Step(step_id=len(steps) + 1, source="agent", model_name="student",
                              timestamp=datetime.fromtimestamp(m["ts"] / 1000).isoformat(),
                              message=SID.sub("AC_REDACTED_SID", "\n".join(m["text"])),
                              tool_calls=calls or None, observation=Observation(results=res) if res else None))
        sid = next((e["sessionID"] for e in events if "sessionID" in e), str(uuid.uuid4()))
        return Trajectory(session_id=sid, agent=Agent(name="altimate-code", version="0.12.8", model_name="student"),
                          steps=steps, final_metrics=FinalMetrics(total_prompt_tokens=tin, total_completion_tokens=tout,
                                                                  total_steps=len(steps)))

    for cfg in CONFIGS:
        for split in ["train", "test"]:
            name = f"qwen3.6-harness-{args.tag}-{cfg}-{split}"
            out = args.out / name
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            ts = trials(args.jobs, cfg, split)
            locks, results, first = [], [], None
            for task, attempt, t in ts:
                dst = out / t.name
                shutil.copytree(t, dst, symlinks=False)
                if CONFIGS[cfg][0] == "altimate-code" and not (dst / "agent/trajectory.json").exists():
                    (dst / "agent/trajectory.json").write_text(alt_atif(task, t).model_dump_json(indent=2, exclude_none=True))
                job = t.parent
                lock = JobLock.model_validate_json((job / "lock.json").read_text())
                locks += lock.trials
                first = first or (lock, JobConfig.model_validate_json((job / "config.json").read_text()))
                results.append(TrialResult.model_validate_json((t / "result.json").read_text()))
            lock, conf = first
            conf.job_name, conf.jobs_dir = name, args.out
            conf.tasks = [tc for task, _, _ in ts for tc in [type(conf.tasks[0])(path=H / f"tasks/{task}")]]
            lock.trials = locks
            (out / "lock.json").write_text(lock.model_dump_json(indent=4, exclude_none=True))
            (out / "config.json").write_text(conf.model_dump_json(indent=4))
            starts = [r.started_at for r in results if r.started_at]
            ends = [r.finished_at for r in results if r.finished_at]
            jr = JobResult(id=uuid.uuid4(), started_at=min(starts), finished_at=max(ends), updated_at=max(ends),
                           n_total_trials=len(results),
                           stats=JobStats.from_trial_results(results, n_total_trials=len(results)))
            (out / "result.json").write_text(jr.model_dump_json(indent=4))
            print(f"{name}: {len(results)} trials, passed {sum((r.verifier_result.rewards or {}).get('reward') == 1 for r in results if r.verifier_result)}")


def hf(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    test = {(t["source"], t["task"]) for t in json.load(open(H.parent / "data/test_tasks_v3.json"))}
    for cfg, (harness, thinking) in CONFIGS.items():
        rows = []
        # runs after the environment fixes: records.py's per-trial strict reward and termination
        rec = args.jobs / cfg / "records.jsonl"
        recs = {Path(r["trial_dir"]).name: r for r in map(json.loads, rec.open())} if rec.exists() else {}
        for split in ["train", "test"]:
            for task, attempt, t in trials(args.jobs, cfg, split):
                res = json.loads((t / "result.json").read_text())
                reward = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
                msgs = to_messages(cfg, task, t)
                src, tid = task.split("__", 1)
                assert ((src, tid) in test) == (split == "test")
                exc = (res.get("exception_info") or {}).get("exception_type")
                start, end = res.get("started_at"), res.get("finished_at")
                rows.append({"messages": msgs, "tools": "[]", "source": f"qwen3.6-35b-a3b_{cfg}_{args.tag}/{src}/{tid}",
                             "metadata": {"benchmark": src, "task": tid, "attempt": attempt, "split": split,
                                          "reward": float(reward or 0), "success": reward == 1,
                                          "strict": recs[t.name]["strict"] if recs else None,
                                          "stop": recs[t.name]["termination"] if recs else termination(cfg, t),
                                          "exception": exc,
                                          "num_turns": sum(m["role"] == "assistant" for m in msgs),
                                          "seconds": int((datetime.fromisoformat(end) - datetime.fromisoformat(start))
                                                         .total_seconds()) if start and end else None,
                                          "model": MODEL, "harness": harness, "enable_thinking": thinking,
                                          "max_turns": 100, "run": f"harness-comparison-{args.tag}"}})
        d = args.out / f"data/qwen3.6-35b-a3b_{cfg}_{args.tag}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), d / "train.parquet")
        print(f"{d.name}: {len(rows)} rows ({sum(r['metadata']['split'] == 'test' for r in rows)} test-split), "
              f"{sum(r['metadata']['success'] for r in rows)} passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["hub", "hf"])
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"hub": hub, "hf": hf}[a.what](a)
