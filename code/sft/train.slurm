#!/bin/bash
# Submit: sbatch sft/train.slurm data/sft/train.jsonl checkpoints/sft-v1 [extra train.py args]
#SBATCH -J sft-dbt
#SBATCH -A MCB25074
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o /scratch/09059/xliaoyi/post-train-llms/logs/sft-%j.out

cd /scratch/09059/xliaoyi/post-train-llms
DATA=$1 OUT=$2
shift 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.hf
CUDA_VISIBLE_DEVICES=0 .venv/bin/python sft/train.py --data "$DATA" --out "$OUT" "$@"
