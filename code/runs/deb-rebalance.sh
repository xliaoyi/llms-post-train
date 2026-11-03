#!/bin/bash
# When either half finishes, stop the other half's runner and split its remaining attempts across both servers
# (8006 GPU1, 8007 GPU2). Finished (task, attempt) runs are skipped on resume. Appends TOTAL to the slow half's log
# at the end so the waiting pipelines proceed.
cd /scratch/09059/xliaoyi/post-train-llms
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
until grep -q TOTAL runs/qwen35-train-deb-a.log || grep -q TOTAL runs/qwen35-train-deb-b.log; do sleep 60; done
if grep -q TOTAL runs/qwen35-train-deb-a.log; then S=b PID=122664; else S=a PID=122663; fi
date; echo "other half done; stopping half $S runner $PID"
kill $PID; while kill -0 $PID 2>/dev/null; do sleep 5; done
python3 - $S <<'PY'
import json, glob, sys
s = sys.argv[1]
tasks = json.load(open(f"data/new_train_tasks_deb_{s}.json"))
left = {t["task"]: 65 - len(glob.glob(f"runs/qwen35-train/{t['source']}/{t['task']}/*/trajectory.json")) for t in tasks}
halves = [[], []]; load = [0, 0]
for t in sorted(tasks, key=lambda t: -left[t["task"]]):  # greedy balance by remaining attempts
    i = load.index(min(load)); halves[i].append(t); load[i] += left[t["task"]]
for i, h in enumerate(halves):
    json.dump(h, open(f"data/new_train_tasks_deb_{s}{i + 1}.json", "w"), indent=1)
print("remaining per server:", load)
PY
cd harness
for i in 1 2; do
  ../.venv/bin/python agent.py --base-url http://localhost:800$((5 + i))/v1 --model student --tasks-file ../data/new_train_tasks_deb_$S$i.json \
    --attempts 65 --max-turns 100 -j 30 --out ../runs/qwen35-train > ../runs/qwen35-train-deb-$S$i.log 2>&1 &
done
wait; cd ..
tail -qn1 runs/qwen35-train-deb-${S}1.log runs/qwen35-train-deb-${S}2.log
echo "TOTAL (rebalanced; see qwen35-train-deb-${S}1.log / -${S}2.log)" >> runs/qwen35-train-deb-$S.log
date; echo done
