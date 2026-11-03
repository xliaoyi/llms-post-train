#!/bin/bash
# After the 44 new train tasks' rollouts finish: free GPUs 1-2, build SFT data v6, self-distill with train/test eval + wandb.
cd /scratch/09059/xliaoyi/post-train-llms
export LD_LIBRARY_PATH=/opt/apps/gcc/13.2.0/lib64:/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH HF_HOME=$PWD/.hf
S=.hf/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
GPU1=GPU-9949a2f5-5212-36a4-606e-0801007d0ea0 GPU2=GPU-fe5d9f72-7187-e8d8-93ae-699d98af8ab4
until [ -f runs/qwen35-train-deb.DONE ]; do sleep 60; done
date; echo "rollouts done, stopping rollout servers"
PIDS=$(pgrep -f "vllm serve .*--port 800[67] ")
kill $PIDS; while pgrep -f "vllm serve .*--port 800[67] " >/dev/null; do sleep 10; done; sleep 30
nvidia-smi --query-gpu=uuid,memory.used --format=csv
.venv/bin/python sft/prepare.py --runs runs/qwen35-train --model $S --out data/sft/v6-qwen35-self.jsonl || exit 1
BASE=$(python3 -c "
import json,glob; r=[json.load(open(f))['result'].get('reward')==1 for f in glob.glob('runs/qwen35-train/*/*/*/trajectory.json')]
print(round(sum(r)/len(r),4))")
echo "train baseline $BASE"
CUDA_VISIBLE_DEVICES=$GPU1 nohup .venv/bin/python sft/train.py --data data/sft/v6-qwen35-self.jsonl --out checkpoints/sft-v6-qwen35-self \
  --model $S --epochs 5 --lr 1e-4 --grad-accum 8 --lora-r 64 --max-len 49152 --save-every 25 --wait-eval > runs/train-v6-qwen35-self.log 2>&1 &
TP=$!; echo "train pid $TP"
until [ -f checkpoints/sft-v6-qwen35-self/args.json ]; do sleep 10; done
.venv/bin/python sft/eval_loop.py --run checkpoints/sft-v6-qwen35-self --gpu $GPU2 --port 8008 --train-pid $TP \
  --set train:data/train_tasks_v2.json:1:$BASE --set test:data/test_tasks_v2.json:4:0.15 > runs/eval-loop-v6.log 2>&1
echo "eval loop exited $?"; cat checkpoints/sft-v6-qwen35-self/BEST
