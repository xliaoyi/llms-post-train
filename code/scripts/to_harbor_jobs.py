"""Convert our rollouts (runs/**/trajectory.json) into Harbor job directories for `harbor upload`.

  .venv-harbor/bin/python scripts/to_harbor_jobs.py [job ...]     # default: every job in JOBS

Each run becomes a Harbor trial with an ATIF agent/trajectory.json, result.json and lock.json, built
with Harbor's own models (locks via build_job_lock, stats via JobStats). Rollouts were produced by
harness/env.py + harness/agent.py, not by `harbor run`; rewards are the official task verifiers as run
there (equivalent to Harbor's singularity backend, see scripts/harbor_check.py). Task digests are of
the Harbor-format task dirs. Twilio account SIDs in task data are redacted, as on GitHub.
"""
import json
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.job.lock import build_job_lock
from harbor.models.job.result import JobResult, JobStats
from harbor.publisher.packager import Packager
from harbor.models.trajectories import Agent, Observation, ObservationResult, Step, ToolCall, Trajectory
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "harbor-jobs"
TASK_DIRS = {"deb": ROOT / "harbor-check/tasks", "spider2": ROOT / "datasets/spider2-dbt",
             "ade": ROOT / "datasets/ade-bench"}
AGENT = "llms-post-train-dbt-agent"  # harness/agent.py: bash, read_file, write_file, edit_file
LUNA = ("gpt-5.6-luna", "openai")
OPUS = ("claude-opus-5-5", "anthropic")
EVAL = 3  # attempts per eval task
JOBS = {  # <model>/<split> -> (model, [(run dir, max attempts or None, attempt offset)]); Hub name uses "-"
    "gpt-5.6-luna/train": (LUNA, [("teacher-luna", None, 0),
                                  ("teacher-luna-r2", None, 8)]),  # round 2: 16 more attempts on unsolved tasks
    "gpt-5.6-luna/eval": (LUNA, [("eval/luna-teacher-t100", EVAL, 0)]),
    "claude-opus-5.5/train": (OPUS, [("opus-train", None, 0)]),
    "claude-opus-5.5/eval": (OPUS, [("eval/opus-5-5-t100", EVAL, 0)]),
    "qwen3.8-35b-a3b-zeroshot/eval": (("empero-ai/Qwen3.8-35B-A3B-Distill", None), [("eval/zs-qwen38-35b-t100", 4, 0)]),
    "qwen3.5-35b-a3b-zeroshot/eval": (("Qwen/Qwen3.5-35B-A3B", None), [("eval/zs-qwen35-35b-t100", 4, 0)]),
    "qwen3.5-35b-a3b-zeroshot/train": (("Qwen/Qwen3.5-35B-A3B", None), [("qwen35-train", None, 0)]),  # self-distillation data
    "gemma-4-12b-it-zeroshot/eval": (("google/gemma-4-12B-it", None), [("eval/zs-gemma4-12b-it-t100", EVAL, 0)]),
    "qwen3.8-35b-a3b-lora-sft/eval": (("qwen3.8-35b-a3b-lora-sft-v4", None), [("eval/sft-v4-qwen35b-t100", EVAL, 0)]),
    "gemma-4-12b-it-lora-sft/eval": (("gemma-4-12b-it-lora-sft-v4", None), [("eval/sft-v4-gemma12b-t100", EVAL, 0)]),
}
SID = re.compile(r"AC[0-9a-f]{32}")


def to_atif(messages, tools, model, res, session_id):
    steps, i = [], 0
    while i < len(messages):
        m = messages[i]
        if m["role"] in ("system", "user"):
            steps.append(Step(step_id=len(steps) + 1, source=m["role"], message=m["content"] or ""))
            i += 1
            continue
        assert m["role"] == "assistant", m["role"]
        calls, results = [], []
        for c in m.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"])
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(tool_call_id=c["id"], function_name=c["function"]["name"],
                                  arguments=args if isinstance(args, dict) else {"raw": c["function"]["arguments"]}))
        i += 1
        while i < len(messages) and messages[i]["role"] == "tool":
            results.append(ObservationResult(source_call_id=messages[i].get("tool_call_id"),
                                             content=messages[i]["content"]))
            i += 1
        steps.append(Step(step_id=len(steps) + 1, source="agent", model_name=model, message=m.get("content") or "",
                          tool_calls=calls or None, observation=Observation(results=results) if results else None))
    if not steps:  # the run errored before its messages were saved; ATIF needs at least one step
        steps.append(Step(step_id=1, source="system", message=f"[run errored, no messages recorded] {res.get('error', '')}"))
    u = res.get("usage") or {}
    return Trajectory(session_id=session_id, agent=Agent(name=AGENT, version="1.0", model_name=model,
                                                         tool_definitions=tools),
                      steps=steps,
                      final_metrics=FinalMetrics(total_prompt_tokens=u.get("input"),
                                                 total_completion_tokens=u.get("output"),
                                                 total_cached_tokens=u.get("cached"),
                                                 total_cost_usd=res.get("usd"),
                                                 total_steps=len(steps)),
                      extra={k: res.get(k) for k in ("stop", "turns", "attempt", "source")})


def convert(job_name, model, runs):
    model_name, provider = model
    job_dir = OUT / job_name
    split = job_name.split("/")[1]
    files = [(f, run_dir, max_att, offset) for run_dir, max_att, offset in runs
             for f in sorted((ROOT / "runs" / run_dir).glob("*/*/*/trajectory.json"))]
    job_id = uuid.uuid4()
    agent_cfg = AgentConfig(name=AGENT, model_name=model_name)
    job_config = JobConfig(job_name=job_name.replace("/", "-"), jobs_dir=job_dir.parent, agents=[agent_cfg])
    hashes, trial_configs, trial_results = {}, [], []
    for f, run_dir, max_att, offset in files:
        source, task, att = f.parts[-4:-1]
        if max_att is not None and int(att) >= max_att:
            continue
        text = SID.sub("AC_REDACTED_SID", f.read_text())
        d = json.loads(text)
        res = d["result"]
        if res.get("stop") == "budget":
            continue
        task_dir = TASK_DIRS[source] / task
        if task_dir not in hashes:
            hashes[task_dir] = Packager.compute_content_hash(task_dir)[0]
        trial_name = f"{task}__a{int(att) + offset}"
        tc = TrialConfig(task=TaskConfig(path=task_dir), trial_name=trial_name, trials_dir=job_dir,
                         agent=agent_cfg, job_id=job_id)
        trial_dir = job_dir / trial_name
        (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
        tr_id = uuid.uuid4()
        (trial_dir / "agent/trajectory.json").write_text(
            to_atif(d["messages"], d["tools"], model_name, res, str(tr_id)).model_dump_json(indent=1, exclude_none=True))
        finished = datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)
        started = finished - timedelta(seconds=res.get("sec") or 0)
        u = res.get("usage") or {}
        exc = None
        if "error" in res:
            exc = ExceptionInfo(exception_type=res["error"].strip().splitlines()[-1].split(":")[0][:100],
                                exception_message=res["error"].strip().splitlines()[-1][:1000],
                                exception_traceback=res["error"], occurred_at=finished)
        tr = TrialResult(id=tr_id, task_name=task, trial_name=trial_name, trial_uri=trial_dir.as_uri(),
                         task_id=tc.task.get_task_id(), task_checksum=hashes[task_dir], config=tc,
                         agent_info=AgentInfo(name=AGENT, version="1.0",
                                              model_info=ModelInfo(name=model_name, provider=provider)),
                         verifier_result=VerifierResult(rewards={"reward": float(res.get("reward") or 0)}),
                         exception_info=exc, started_at=started, finished_at=finished)
        tr.agent_result = AgentContext(n_input_tokens=u.get("input"), n_cache_tokens=u.get("cached"),
                                       n_output_tokens=u.get("output"), cost_usd=res.get("usd"),
                                       metadata={"split": split, "stop": res.get("stop"), "turns": res.get("turns"),
                                                 "imported_from": f"runs/{run_dir}/{source}/{task}/{att}"})
        (trial_dir / "result.json").write_text(tr.model_dump_json(indent=4))
        (trial_dir / "config.json").write_text(tc.model_dump_json(indent=4))
        trial_configs.append(tc)
        trial_results.append(tr)
    downloads = {tc.task.get_task_id(): SimpleNamespace(path=tc.task.path, content_hash=None,
                                                        resolved_git_commit_id=None) for tc in trial_configs}
    lock = build_job_lock(config=job_config, trial_configs=trial_configs, task_download_results=downloads)
    for tc, tl in zip(trial_configs, lock.trials):
        (job_dir / tc.trial_name / "lock.json").write_text(tl.model_dump_json(indent=4, exclude_none=True))
    (job_dir / "lock.json").write_text(lock.model_dump_json(indent=4, exclude_none=True))
    (job_dir / "config.json").write_text(job_config.model_dump_json(indent=4))
    starts = [t.started_at for t in trial_results]
    ends = [t.finished_at for t in trial_results]
    jr = JobResult(id=job_id, started_at=min(starts), finished_at=max(ends), updated_at=max(ends),
                   n_total_trials=len(trial_results),
                   stats=JobStats.from_trial_results(trial_results, n_total_trials=len(trial_results)))
    (job_dir / "result.json").write_text(jr.model_dump_json(indent=4))
    passed = sum(t.verifier_result.rewards["reward"] == 1 for t in trial_results)
    print(f"{job_name}: {len(trial_results)} trials, {passed} passed")


if __name__ == "__main__":
    for name in sys.argv[1:] or JOBS:
        convert(name, *JOBS[name])
