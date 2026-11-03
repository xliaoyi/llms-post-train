#!/bin/bash
# v6b: self-distill on all 1,383 passes (<=127k tokens), layers split over GPUs 1+2; checkpoints evaluated after training.
cd /scratch/09059/xliaoyi/post-train-llms
export LD_LIBRARY_PATH=/opt/apps/gcc/13.2.0/lib64:/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH HF_HOME=$PWD/.hf
S=.hf/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
GPU1=GPU-9949a2f5-5212-36a4-606e-0801007d0ea0 GPU2=GPU-fe5d9f72-7187-e8d8-93ae-699d98af8ab4
R=checkpoints/sft-v6b-qwen35-self
CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 nohup .venv/bin/python sft/train.py --data data/sft/v6b-qwen35-self.jsonl --out $R \
  --model $S --epochs 1 --lr 1e-4 --grad-accum 8 --lora-r 64 --save-every 25 > runs/train-v6b-qwen35-self.log 2>&1 &
TP=$!; echo "train pid $TP"
# the end-of-epoch checkpoint (step 173) is saved as epoch1; name it step<N> so the eval loop picks it up
(while kill -0 $TP 2>/dev/null; do sleep 30; done
 N=$(tail -1 $R/train_log.jsonl | python3 -c "import json,sys; print(json.load(sys.stdin)['step'])")
 [ -d $R/epoch1 ] && [ ! -d $R/step$N ] && mv $R/epoch1 $R/step$N && echo "epoch1 -> step$N") &
until [ -f $R/args.json ]; do sleep 10; done
.venv/bin/python sft/eval_loop.py --run $R --gpu $GPU1 $GPU2 --port 8008 8009 --train-pid $TP --after-train \
  --set train:data/train_tasks_v2.json:1:0.185 --set test:data/test_tasks_v2.json:4:0.15 > runs/eval-loop-v6b.log 2>&1
echo "eval loop exited $?"; cat $R/BEST
