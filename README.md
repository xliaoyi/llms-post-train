# llms-post-train: teacher trajectories for dbt agent distillation

Agent rollouts on Spider2-DBT and ADE-bench training tasks, collected to distill a teacher into a small student.
Every rollout ran in the task's official container image (via Apptainer) with the same system prompt and four tools
(`bash`, `read_file`, `write_file`, `edit_file`), and was graded by the task's own tests (`reward` 1 = pass).
Evaluation (data-eng-bench analytics) trajectories are deliberately **not** included, to avoid leaking test solutions.

## Layout

```
trajectories/<run>/<source>/<task>/<attempt>/trajectory.json
sft/<version>.jsonl          # SFT datasets built from the passing trajectories
```

| run | teacher | rollouts | passed |
|---|---|---|---|
| teacher-pilot, teacher-pilot-reason-med | gpt-5.6-luna | 20 | 3 |
| teacher-luna | gpt-5.6-luna | 568 | 106 |
| teacher-luna-r2 | gpt-5.6-luna (unsolved tasks) | 800 | 10 |
| claude-pilot-opus-5-5 | claude-opus-5-5 | 20 | 9 |
| claude-pilot-sonnet-5-5 | claude-sonnet-5-5 | 20 | 4 |
| claude-coverage | claude-opus-5-5 | 148 | 33 |
| dagger-v2 | Qwen3.5-2B SFT student, then claude-opus-5-5 takes over | 142 | 63 |

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

All teachers are stored in this one provider-neutral chat format (Claude's `tool_use`/`tool_result` blocks are
converted), so they can be mixed and rendered with any student's chat template. Hidden reasoning/thinking is not
stored. In DAgger runs, the student's own turns carry `"student": true` and are excluded from the SFT loss.

## sft/*.jsonl

SFT datasets built from passing trajectories (`reward == 1`), **already tokenized with the Qwen3.5 chat template**:
`{"source": "<trajectory path>", "input_ids": [...], "labels": [...]}`, where `labels` is -100 everywhere except
non-student assistant turns. To train a different student (e.g. Gemma), rebuild from `trajectories/` with that model's
template. `luna-v1` (Luna round 1), `luna-v2` (+ round 2), `v3` (Luna + Opus coverage + DAgger: 226 examples, 42 tasks).

Note: one Twilio account SID in Spider2's `twilio001` sample data is replaced with `AC_REDACTED_SID`.
