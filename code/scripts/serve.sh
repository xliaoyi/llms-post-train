#!/bin/bash
# Serve a model with vLLM for the agent.
#   bash scripts/serve.sh <gpu> <port> <model_dir> [thinking]
# "thinking" adds the reasoning parser (needed when the agent runs with --thinking).
# MEM=<fraction> overrides --gpu-memory-utilization (e.g. when sharing a GPU).
# TOOL_PARSER overrides qwen3_coder (e.g. gemma4); VLLM_ARGS adds extra vllm flags (e.g. --quantization fp8).
GPU=$1 PORT=$2 MODEL=$3
cd "$(dirname "$0")/.."
EXTRA=""
[ "$4" = "thinking" ] && EXTRA="--reasoning-parser qwen3"
CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$PWD/.hf VLLM_CACHE_ROOT=$PWD/.vllm_cache exec .venv/bin/vllm serve "$MODEL" \
  --served-model-name student --port "$PORT" --max-model-len 131072 --gpu-memory-utilization ${MEM:-0.85} \
  --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER:-qwen3_coder} --enable-prefix-caching $EXTRA $VLLM_ARGS
