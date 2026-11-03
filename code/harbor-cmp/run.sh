#!/bin/bash
# Harness comparison on Qwen3.6-35B-A3B: Qwen Code vs altimate-code, thinking on vs off, 4 attempts on the v3 split.
# Run from the code root (the directory holding harness/, scripts/, harbor-cmp/). Before the first run:
#   python harbor-cmp/prepare_tasks.py; python harbor-cmp/dbt_mirror.py build   (needs internet)
#   oracle runs (validate_trial.sh) + python harbor-cmp/strict/build_strict.py; python harbor-cmp/preflight.py
#   GPU0 vLLM :8008 <- proxies 8100 (qwen-think), 8101 (qwen-nothink)
#   GPU1 vLLM :8009 <- proxies 8102 (alt-think),  8103 (alt-nothink)
export LD_LIBRARY_PATH=/opt/apps/gcc/13.2.0/lib64:/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
S=$(ls -d $PWD/.hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*)
mkdir -p logs
for g in 0 1; do
  MEM=0.85 VLLM_ARGS='--max-model-len 262144 --quantization fp8 --max-num-seqs 64 --limit-mm-per-prompt {"image":0,"video":0}' \
    setsid nohup bash scripts/serve.sh $g 800$((8+g)) $S thinking > logs/vllm-qwen36-gpu$g.log 2>&1 &
done
# Qwen3.6 recommended sampling for each mode; the proxy also sets enable_thinking and drops web tools
ON="--thinking on --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0"
OFF="--thinking off --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5"
setsid nohup .venv/bin/python harness/proxy.py --log-dir harbor-cmp/proxy-logs --port 8100 --backend http://localhost:8008 $ON > logs/proxy-8100.log 2>&1 &
setsid nohup .venv/bin/python harness/proxy.py --log-dir harbor-cmp/proxy-logs --port 8101 --backend http://localhost:8008 $OFF > logs/proxy-8101.log 2>&1 &
setsid nohup .venv/bin/python harness/proxy.py --log-dir harbor-cmp/proxy-logs --port 8102 --backend http://localhost:8009 $ON > logs/proxy-8102.log 2>&1 &
setsid nohup .venv/bin/python harness/proxy.py --log-dir harbor-cmp/proxy-logs --port 8103 --backend http://localhost:8009 $OFF > logs/proxy-8103.log 2>&1 &
# offline mirror of the dbt packages (the agents have no internet; run_trial.sh points dbt deps at it)
setsid nohup python3 harbor-cmp/dbt_mirror.py serve > logs/dbt-mirror.log 2>&1 &
until curl -sf localhost:8008/v1/models > /dev/null && curl -sf localhost:8009/v1/models > /dev/null; do sleep 30; done
for c in qwen-think qwen-nothink alt-think alt-nothink; do
  setsid nohup .venv/bin/python harbor-cmp/driver.py --config $c -j 12 > harbor-cmp/driver-$c.log 2>&1 &
done
