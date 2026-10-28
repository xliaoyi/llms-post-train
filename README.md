# llms-post-train: teacher trajectories for dbt agent distillation

Agent rollouts of two teacher models, **gpt-5.6-luna** and **claude-opus-5-5**, on dbt analytics tasks.
Training tasks come from Spider2-DBT and ADE-bench; evaluation tasks are the analytics category of data-eng-bench
(DuckDB backend). Every rollout ran in the task's official container image (via Apptainer) with the same system
prompt and four tools (`bash`, `read_file`, `write_file`, `edit_file`), and was graded by the task's own tests
(`reward` 1 = pass).

## Results

| Dataset | Split | Tasks in total | Tasks used | Attempts per task (Luna / Opus) | Luna 5.6: rollouts passed | Luna 5.6: tasks solved | Opus 5.5: rollouts passed | Opus 5.5: tasks solved |
|---|---|---|---|---|---|---|---|---|
| Spider2-DBT | train | 64 | 55 | 8 / 5 | 69/440 (15.7%) | 14/55 | 127/275 (46.2%) | 28/55 |
| ADE-bench | train | 75 | 16 | 8 / 5 | 37/128 (28.9%) | 7/16 | 46/80 (57.5%) | 13/16 |
| data-eng-bench (analytics) | eval | 103 | 64 | 3 / 3 | 38/192 (19.8%) | 21/64 | 130/192 (67.7%) | 48/64 |

- *Tasks used*: Spider2-DBT drops 9 tasks whose reference solution fails our oracle check; ADE-bench keeps only
  analytics-style tasks (model creation, metrics, grain fixes), dropping debugging, refactoring, config/package and
  snapshot tasks; data-eng-bench keeps the 64 tasks labelled `analytics`. Two of those
  (`dbt-product-return-rate-analysis`, `dbt-fix-marketing-attribution`) have broken reference solutions, so 62/64 is the ceiling.
- *Tasks solved* = solved in at least one attempt.
- Luna train numbers are round 1 only (8 attempts on every task). Luna's round 2 added 16 attempts on its 50 unsolved
  training tasks (10/800 passed); with all rounds and pilots Luna solves 16/55 Spider2 and 8/16 ADE tasks.
- Max turns: 40 on training tasks, 100 on eval tasks. Opus runs use adaptive thinking with effort `medium`;
  thinking content is not stored.

## Harbor Hub

The same rollouts, plus the student evaluations, are public on [Harbor Hub](https://hub.harborframework.com) as
one job per model and split. They were converted from our `trajectory.json` files to Harbor job format (ATIF
trajectories); rewards are the task verifiers as run by our Apptainer runner, which gives the same oracle (62/64)
and empty-solution (0/64) results as Harbor's own singularity backend on the 64 eval tasks.

| Model | Split | Trials | Passed | Harbor Hub job |
|---|---|---|---|---|
| gpt-5.6-luna | train | 1368 | 116 | [link](https://hub.harborframework.com/jobs/1f791ffe-d34a-4f34-8b9d-85672e91c655) |
| gpt-5.6-luna | eval | 192 | 38 | [link](https://hub.harborframework.com/jobs/73320754-56dc-4db1-9c15-fcbf007c0ec5) |
| claude-opus-5.5 | train | 355 | 173 | [link](https://hub.harborframework.com/jobs/280a0bbf-8a9c-448b-946a-7756d0d32e7a) |
| claude-opus-5.5 | eval | 192 | 130 | [link](https://hub.harborframework.com/jobs/6fd67471-a37f-4076-97ee-f73747f12686) |
| Qwen3.8-35B-A3B-Distill, zero-shot | eval | 192 | 27 | [link](https://hub.harborframework.com/jobs/673854dd-bb4b-4418-bbac-52d77619fa31) |
| gemma-4-12B-it, zero-shot | eval | 192 | 9 | [link](https://hub.harborframework.com/jobs/37603119-d5c8-4c6d-a596-7a3b218af0cc) |
| Qwen3.8-35B-A3B-Distill, LoRA SFT | eval | 192 | 5 | [link](https://hub.harborframework.com/jobs/2c1f2932-f62a-4e84-a5d2-25b7cb97c167) |

Luna train = round 1 (attempts 0–7) and round 2 (attempts 8–23 on the 50 tasks round 1 did not solve); pilots are
not included.

## Layout

```
trajectories/
  gpt-5.6-luna/
    train/round1/                   71 tasks x 8 attempts   (568)
    train/round2/                   50 unsolved tasks x 16  (800)
    train/pilot/, train/pilot-reasoning-medium/   10 + 10 pilot rollouts
    eval/                           64 tasks x 3 attempts   (192)
  opus-5.5/
    train/                          71 tasks x 5 attempts   (355)
    eval/                           64 tasks x 3 attempts   (192)
each: <source>/<task>/<attempt>/trajectory.json      (source: spider2 | ade | deb)
sft/<version>.jsonl                 SFT datasets (see below)
```

## trajectory.json

```json
{"result": {"source": "...", "task": "...", "attempt": 0, "reward": 1, "stop": "finished", "turns": 23, "usage": {...}},
 "tools": [ ...OpenAI-style function schemas... ],
 "messages": [
   {"role": "system", "content": "..."},
   {"role": "user", "content": "<task instruction>"},
   {"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "type": "function",
       "function": {"name": "bash", "arguments": "{\"command\": \"dbt run\"}"}}]},
   {"role": "tool", "tool_call_id": "...", "content": "<tool output>"}
 ]}
```

Both teachers are stored in this one provider-neutral chat format (Claude's `tool_use`/`tool_result` blocks are
converted), so they can be mixed and rendered with any student's chat template. Hidden reasoning is not stored.

## sft/*.jsonl

SFT datasets built from passing training trajectories (`reward == 1`; never from eval), **already tokenized with the
Qwen3.5 chat template**: `{"source": "<trajectory path>", "input_ids": [...], "labels": [...]}`, where `labels` is -100
everywhere except teacher assistant turns. To train a different student (e.g. Gemma), rebuild from `trajectories/`
with that model's template. `luna-v1` (Luna round 1), `luna-v2` (+ round 2), `v3` (Luna + Opus coverage + DAgger
rollouts, where a Qwen3.5-2B student acted first and Opus took over; the DAgger rollouts are not included here).

Note: one Twilio account SID in Spider2's `twilio001` sample data is replaced with `AC_REDACTED_SID`.
