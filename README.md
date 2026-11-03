# llms-post-train: teacher trajectories for dbt agent distillation

Agent rollouts of two teacher models, **gpt-5.6-luna** and **claude-opus-5-5**, on dbt analytics tasks, plus
evaluation rollouts of the open-weight students (zero-shot and after LoRA SFT on the teacher trajectories) and the
code that produced them (`code/`).
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

### Students (data-eng-bench analytics, eval)

| Model | Attempts per task | Rollouts passed | Tasks solved |
|---|---|---|---|
| Qwen3.5-35B-A3B, zero-shot | 4 | 38/256 (14.8%) | 25/64 |
| Qwen3.8-35B-A3B-Distill, zero-shot | 4 | 32/256 (12.5%) | 19/64 |
| gemma-4-12B-it, zero-shot | 3 | 9/192 (4.7%) | 5/64 |
| Qwen3.8-35B-A3B-Distill, LoRA SFT (v4) | 3 | 5/192 (2.6%) | 4/64 |
| gemma-4-12B-it, LoRA SFT (v4) | 3 | 0/192 (0.0%) | 0/64 |

Students were served with vLLM and ran in the same harness, prompt, tools and 100-turn limit as the teachers.

### Qwen3.5-35B-A3B self-distillation (v6b), per-checkpoint evaluation

The 64 data-eng-bench analytics tasks are re-split into 44 that join the training pool and 20 held-out test tasks
(`code/data/train_tasks_v2.json`: 55 Spider2 + 16 ADE + 44 deb = 115 tasks; `code/data/test_tasks_v2.json`: 20 deb tasks).
v6b is a LoRA (r 64, lr 1e-4, 1 epoch, 172 steps) on all 1,383 passing Qwen3.5-35B-A3B rollouts within 127k tokens;
every 25-step checkpoint and the final one was evaluated on train (1 attempt per task) and test (4 attempts per task).
SR@1 = solved on attempt 0, SR@4 = solved in any of 4 attempts, avg = rollouts passed / rollouts.

| Checkpoint | Train pass rate (115) | Test SR@1 | Test SR@4 | Test avg (80) |
|---|---|---|---|---|
| base (zero-shot) | 18.5% | 15.0% | 40.0% | 15.0% (12/80) |
| step25 | 20.0% | 5.0% | 20.0% | 12.5% (10/80) |
| step50 | 11.3% | 0.0% | 30.0% | 12.5% (10/80) |
| step75 | 13.9% | 10.0% | 30.0% | 11.2% (9/80) |
| step100 | 23.5% | 5.0% | 20.0% | 6.2% (5/80) |
| step125 | 19.1% | 5.0% | 30.0% | 11.2% (9/80) |
| step150 | 17.4% | 20.0% | 30.0% | 11.2% (9/80) |
| step172 | 21.7% | 15.0% | 20.0% | 10.0% (8/80) |

Base test numbers are the zero-shot eval rollouts restricted to the 20 test tasks; the base train rate (18.5%) is
the baseline used by the eval loop. No checkpoint beats the base model on the test tasks.

### Task split v3 (105 train / 30 test)

Used from the Qwen3.6 harness comparison onward: the 20 v2 test tasks plus 10 more data-eng-bench analytics tasks
(drawn with a fixed seed from the 44 v2 training ones) form a 30-task test set; the other 34 data-eng-bench tasks join
the 55 Spider2-DBT and 16 ADE-bench training tasks. `code/data/train_tasks_v3.json` (105) and
`code/data/test_tasks_v3.json` (30) list them as `{"source", "task"}`.

### Qwen3.6-35B-A3B harness comparison (v1, before environment fixes)

Qwen3.6-35B-A3B (vLLM, FP8) in two off-the-shelf agent harnesses, each with thinking on and off, run through
Harbor 0.25.0 (singularity backend) on the v3 split: 105 train and 30 test tasks, 4 attempts each.
Qwen Code 0.26.5 (Harbor's built-in agent) and altimate-code 0.12.8 (our Harbor adapter,
`code/harbor-cmp/altimate_agent.py`); every model call goes through `code/harness/proxy.py`, which sets the sampling
(thinking on: temperature 1.0, top-p 0.95, top-k 20; off: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5)
and removes web tools. At most 100 turns, 2-hour agent timeout. Rewards are the benchmarks' own tests.
SR@1 = solved on attempt 0, SR@4 = solved in any of 4 attempts, avg = rollouts passed / rollouts.

| Config | Train avg | Train SR@1 | Train SR@4 | Test avg | Test SR@1 | Test SR@4 |
|---|---|---|---|---|---|---|
| Qwen Code, thinking on | 23.6% (99/420) | 21.9% (23/105) | 42.9% (45/105) | 22.5% (27/120) | 30.0% (9/30) | 46.7% (14/30) |
| Qwen Code, thinking off | 19.3% (81/420) | 16.2% (17/105) | 32.4% (34/105) | 17.5% (21/120) | 23.3% (7/30) | 43.3% (13/30) |
| altimate-code, thinking on | 26.9% (113/420) | 26.7% (28/105) | 43.8% (46/105) | 24.2% (29/120) | 30.0% (9/30) | 46.7% (14/30) |
| altimate-code, thinking off | 19.0% (80/420) | 17.1% (18/105) | 32.4% (34/105) | 15.0% (18/120) | 13.3% (4/30) | 36.7% (11/30) |

This run predates the environment fixes (next commit): agents had internet access, reference solutions were not
validated (9 ADE-bench solutions used a wrong path, 2 data-eng-bench tasks were broken), trial workspaces were not
anonymous, and there was no stricter semantic grading. The rerun with the fixes is v2 below.

Environment fixes (`code/harbor-cmp`, used by the rerun):
- **Validated tasks:** `preflight.py` checks each task's setup in its image, runs the reference solution twice
  (must pass) and an empty solution (must fail); 134/135 tasks pass. Fixed the ADE-bench solution path and the
  `dbt-fix-marketing-attribution` grader schema; `dbt-product-return-rate-analysis` is excluded (its instructions
  contradict its grader, `excluded_tasks.json`). The driver runs only validated tasks (`validated_tasks.json`).
- **Stricter grading:** `strict/` adds a second check to each verifier, reported next to the original reward:
  data-eng-bench outputs are compared with the reference solution's tables (`build_strict.py` from two oracle runs),
  Spider2 with an added row-alignment check. Mutation test (`strict/mutate.py`): of 19 one-line mistakes that change
  the output, the original graders catch 11, the strict check 19.
- **No internet:** agents run with a dead HTTP proxy, offline pip/npm, and a local mirror of the dbt packages
  (`dbt_mirror.py`); the agents are installed from read-only local copies with pinned versions (`qwen_agent.py`,
  Qwen Code 0.26.5; altimate-code 0.12.8).
- **Isolated trials:** each trial gets an anonymous random workspace path.
- **Faithful records:** the proxy logs every model request/response per trial; trials failing for infrastructure
  reasons are kept apart, not deleted; `records.py` writes per-trial termination (completed, turn limit, timeout,
  API error, crash) and metrics on both rewards. Qwen Code's stream idle timeout is disabled.

Trajectories: Hugging Face [`xliaoyi/dbt-agent-sft`](https://huggingface.co/datasets/xliaoyi/dbt-agent-sft), subsets
`qwen3.6-35b-a3b_{qwen-think,qwen-nothink,alt-think,alt-nothink}_v1` (540 rows each, `metadata.split`).
Full Harbor jobs (logs, verifier output; altimate-code trajectories converted to ATIF) on Harbor Hub:

| Config | Train (420 trials) | Test (120 trials) |
|---|---|---|
| Qwen Code, thinking on | [link](https://hub.harborframework.com/jobs/b7798028-e462-41bf-887c-ddd528138bcc) | [link](https://hub.harborframework.com/jobs/0dc11843-6e46-4ac5-85ce-5282d6f42aba) |
| Qwen Code, thinking off | [link](https://hub.harborframework.com/jobs/368bc8af-8fb5-48f8-97fc-e2daa751ed27) | [link](https://hub.harborframework.com/jobs/4530245c-4bfb-4663-a51a-843d1b5a9862) |
| altimate-code, thinking on | [link](https://hub.harborframework.com/jobs/f734fb64-526e-4c92-a78b-619699242c48) | [link](https://hub.harborframework.com/jobs/c24cf2e2-4ae6-454a-983b-4e225217dfa5) |
| altimate-code, thinking off | [link](https://hub.harborframework.com/jobs/f17d9976-075f-4011-a82e-e6bc3cfc69c4) | [link](https://hub.harborframework.com/jobs/2665f584-d09d-4db8-960c-03c67f4f8983) |

### Qwen3.6-35B-A3B harness comparison (v2, after environment fixes)

The same 4 configs rerun with the environment fixes listed above, on the 134 validated tasks (105 train, 29 test;
`dbt-product-return-rate-analysis` excluded), 4 attempts each. *Reward* is each benchmark's own test; *strict* also
requires the strict semantic check (`code/harbor-cmp/strict/`) to pass.

Reward:

| Config | Train avg | Train SR@1 | Train SR@4 | Test avg | Test SR@1 | Test SR@4 |
|---|---|---|---|---|---|---|
| Qwen Code, thinking on | 25.2% (106/420) | 27.6% (29/105) | 41.9% (44/105) | 19.0% (22/116) | 27.6% (8/29) | 51.7% (15/29) |
| Qwen Code, thinking off | 24.0% (101/420) | 22.9% (24/105) | 43.8% (46/105) | 20.7% (24/116) | 17.2% (5/29) | 34.5% (10/29) |
| altimate-code, thinking on | 26.9% (113/420) | 28.6% (30/105) | 42.9% (45/105) | 22.4% (26/116) | 10.3% (3/29) | 48.3% (14/29) |
| altimate-code, thinking off | 20.5% (86/420) | 19.0% (20/105) | 38.1% (40/105) | 17.2% (20/116) | 20.7% (6/29) | 31.0% (9/29) |

Strict:

| Config | Train avg | Train SR@1 | Train SR@4 | Test avg | Test SR@1 | Test SR@4 |
|---|---|---|---|---|---|---|
| Qwen Code, thinking on | 19.5% (82/420) | 21.0% (22/105) | 34.3% (36/105) | 11.2% (13/116) | 20.7% (6/29) | 31.0% (9/29) |
| Qwen Code, thinking off | 20.0% (84/420) | 17.1% (18/105) | 35.2% (37/105) | 13.8% (16/116) | 13.8% (4/29) | 24.1% (7/29) |
| altimate-code, thinking on | 22.6% (95/420) | 22.9% (24/105) | 35.2% (37/105) | 15.5% (18/116) | 3.4% (1/29) | 31.0% (9/29) |
| altimate-code, thinking off | 17.9% (75/420) | 16.2% (17/105) | 30.5% (32/105) | 13.8% (16/116) | 17.2% (5/29) | 24.1% (7/29) |

Termination (536 trials per config): Qwen Code thinking on 494 completed / 42 turn limit; off 460 / 76;
altimate-code thinking on 402 completed / 117 turn limit / 10 context overflow / 7 timeout; off 363 / 163 / 3 / 7.
No model API errors. Per-config numbers: `code/harbor-cmp/results_v2.json` (v1: `results_v1.json`).

Trajectories: Hugging Face [`xliaoyi/dbt-agent-sft`](https://huggingface.co/datasets/xliaoyi/dbt-agent-sft), subsets
`qwen3.6-35b-a3b_{qwen-think,qwen-nothink,alt-think,alt-nothink}_v2` (536 rows each; `metadata.reward`,
`metadata.strict`, `metadata.stop`). Harbor Hub jobs (shared with `inf385t-llms-post-train`):

| Config | Train (420 trials) | Test (116 trials) |
|---|---|---|
| Qwen Code, thinking on | [link](https://hub.harborframework.com/jobs/6dc19973-c9c0-412d-bf81-3f0fa4e67cfe) | [link](https://hub.harborframework.com/jobs/7e30c7e3-e8fb-4301-98d8-20468bc9ba68) |
| Qwen Code, thinking off | [link](https://hub.harborframework.com/jobs/290a5c2f-80e0-4709-bde8-e66dafe24c8a) | [link](https://hub.harborframework.com/jobs/1885915c-dea1-456e-bf7a-0ab9e4f581f8) |
| altimate-code, thinking on | [link](https://hub.harborframework.com/jobs/a40300f5-196a-4c29-8880-fbd25f3e6f4c) | [link](https://hub.harborframework.com/jobs/da02ce00-a3d8-473a-9073-3d97694c93a9) |
| altimate-code, thinking off | [link](https://hub.harborframework.com/jobs/41f734c5-bc03-4949-ab0f-8eb847a4d672) | [link](https://hub.harborframework.com/jobs/a1c619e7-026b-4e8d-b6fb-dc048528b333) |

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
| Qwen3.5-35B-A3B, zero-shot | train | 4720 | 1012 | [link](https://hub.harborframework.com/jobs/21501549-eb99-4d61-b330-a917c2fa3178) |
| Qwen3.5-35B-A3B, zero-shot | eval | 256 | 38 | [link](https://hub.harborframework.com/jobs/28fa0a97-389e-4fc2-9676-6652df770eb8) |
| Qwen3.8-35B-A3B-Distill, zero-shot | eval | 256 | 32 | [link](https://hub.harborframework.com/jobs/78f73469-ba98-4b13-97da-322875e2bcc5) |
| gemma-4-12B-it, zero-shot | eval | 192 | 9 | [link](https://hub.harborframework.com/jobs/37603119-d5c8-4c6d-a596-7a3b218af0cc) |
| Qwen3.8-35B-A3B-Distill, LoRA SFT | eval | 192 | 5 | [link](https://hub.harborframework.com/jobs/2c1f2932-f62a-4e84-a5d2-25b7cb97c167) |
| gemma-4-12B-it, LoRA SFT | eval | 192 | 0 | [link](https://hub.harborframework.com/jobs/c1b095a1-4d51-42fd-a131-57ad921de9c2) |

Luna train = round 1 (attempts 0–7) and round 2 (attempts 8–23 on the 50 tasks round 1 did not solve); pilots are
not included. An earlier 3-attempt Qwen3.8 zero-shot job
([673854dd](https://hub.harborframework.com/jobs/673854dd-bb4b-4418-bbac-52d77619fa31), 27/192) is superseded by the
4-attempt one above.

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
  qwen3.5-35b-a3b-zeroshot/train/           71 x 65, + 3 more on the 35 solved tasks  (4720)
  qwen3.5-35b-a3b-zeroshot/eval/            64 x 4  (256)
  qwen3.8-35b-a3b-distill-zeroshot/eval/    64 x 4  (256)
  gemma-4-12b-it-zeroshot/eval/             64 x 3  (192)
  qwen3.8-35b-a3b-distill-lora-sft/eval/    64 x 3  (192)
  gemma-4-12b-it-lora-sft/eval/             64 x 3  (192)
  qwen3.5-35b-a3b-v6b-self-sft/{train,test}/step<N>/   per checkpoint: 115 x 1 train, 20 x 4 test  (1365)
each: <source>/<task>/<attempt>/trajectory.json      (source: spider2 | ade | deb)
sft/<version>.jsonl                 SFT datasets (see below)
code/
  harness/      Apptainer task runner (env.py), 4-tool agent (agent.py), oracle check, DAgger rollouts,
                sampling and logging proxy for third-party agent CLIs (proxy.py)
  sft/          SFT data prep, chunked loss, full / LoRA trainer, SLURM script
  scripts/      vLLM serving, Harbor parity check, conversion to Harbor jobs, Hugging Face export
  sft/eval_loop.py   per-checkpoint train/test evaluation with early stopping and wandb logging
  runs/         launch scripts for the Qwen3.5 train rollouts, HF upload and the v6 / v6b SFT + eval pipelines
  data/         task selection and task lists;  images/defs/  Apptainer image definitions
  patches/      Harbor singularity --fakeroot patch
  harbor-cmp/   Qwen3.6 harness comparison through Harbor: task prep, Harbor env and altimate-code adapter,
                per-trial runner, driver, launch script (run.sh), publishing to Harbor Hub / HF (publish.py)
                task validation (preflight.py), strict grading (strict/), offline dbt mirror, records.py
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
`v5-qwen35-self.jsonl.gz` (gzipped) is self-distillation data: the 1012 passing Qwen3.5-35B-A3B train rollouts, 962 of
them within 65,536 tokens; training uses the 788 within 49,152 tokens (34 tasks).

Note: one Twilio account SID in Spider2's `twilio001` sample data is replaced with `AC_REDACTED_SID`.
