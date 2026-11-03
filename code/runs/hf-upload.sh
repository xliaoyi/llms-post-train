#!/bin/bash
# After the 44 new train tasks' rollouts finish: export runs/qwen35-train to HF (tmax-sft layout) with a dataset card.
cd /scratch/09059/xliaoyi/post-train-llms
export LD_LIBRARY_PATH=/opt/apps/gcc/13.2.0/lib64:/opt/apps/python/3.12.11/lib HF_HOME=$PWD/.hf
REPO=xliaoyi/dbt-agent-sft N=qwen3.5-35b-a3b_self OUT=hf/dbt-agent-sft
until [ -f runs/qwen35-train-deb.DONE ]; do sleep 60; done
.venv/bin/python scripts/export_hf.py --runs runs/qwen35-train --name $N --model Qwen/Qwen3.5-35B-A3B --out $OUT || exit 1
.venv/bin/python - "$N" "$OUT" <<'PY'
import sys, pyarrow.parquet as pq
n, out = sys.argv[1:]
cnt = {s: pq.read_table(f"{out}/data/{n}_{s}/train.parquet", columns=["source"]).column("source").to_pylist()
       for s in ["all", "only_success"]}
yaml = "".join(f"- config_name: {n}_{s}\n  data_files:\n  - split: train\n    path: data/{n}_{s}/train.parquet\n" for s in cnt)
rows = "".join(f"| `{n}_{s}` | {len(v):,} | {len(set(v))} |\n" for s, v in cnt.items())
open(f"{out}/README.md", "w").write(f"""---
license: odc-by
language:
- en
task_categories:
- text-generation
tags:
- agent
- sft
- tool-use
- dbt
- sql
- self-distillation
configs:
{yaml}---

# dbt agent SFT trajectories

Agent trajectories of **Qwen3.5-35B-A3B** (no thinking) solving dbt / SQL data-engineering tasks, collected for
self-distillation in the INF 385T *LLM post-training* course project. Layout follows
[allenai/tmax-sft](https://huggingface.co/datasets/allenai/tmax-sft): `messages`, `tools`, `source`, `metadata`.

| Subset | Rows | Tasks |
|---|---|---|
{rows}
- **Tasks:** 115 train tasks (55 Spider2-DBT, 16 ADE-bench, 44 data-eng-bench analytics); the 20 held-out data-eng-bench test tasks are excluded.
- **Rollouts:** 65 attempts per task, at most 100 turns, served with vLLM (FP8). Reward 1 = the task's tests pass.
- **Tools:** `bash`, `read_file`, `write_file`, `edit_file` (OpenAI function-calling format, JSON in `tools`).
- **`_only_success`:** rollouts with reward 1, the self-distillation SFT data. Training used only examples of at most
  49,152 tokens under the Qwen3.5 chat template.
- **`metadata`:** benchmark, task, attempt, reward, success, stop reason, number of turns, wall-clock seconds, model.
- Twilio account SIDs appearing in tool output are replaced by `AC_REDACTED_SID`.

```python
from datasets import load_dataset
ds = load_dataset("xliaoyi/dbt-agent-sft", "{n}_only_success", split="train")
```

Code, teacher trajectories and evaluation results: https://github.com/xliaoyi/llms-post-train
""")
PY
.venv/bin/python -c "
from huggingface_hub import HfApi; a=HfApi(); a.create_repo('$REPO', repo_type='dataset', exist_ok=True)
a.upload_folder(repo_id='$REPO', repo_type='dataset', folder_path='$OUT', commit_message='Add $N subsets'); print('uploaded https://huggingface.co/datasets/$REPO')"
