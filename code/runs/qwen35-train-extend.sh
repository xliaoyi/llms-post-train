#!/bin/bash
# After the 35-attempt run of half $1 finishes, resume it to 65 attempts (finished runs are skipped).
cd /scratch/09059/xliaoyi/post-train-llms/harness
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
until grep -q TOTAL ../runs/qwen35-train-$1.log; do sleep 60; done
../.venv/bin/python agent.py --base-url http://localhost:$2/v1 --model student --tasks-file ../data/train_tasks_$1.json \
  --attempts 65 --max-turns 100 -j 30 --out ../runs/qwen35-train > ../runs/qwen35-train-$1-r2.log 2>&1
