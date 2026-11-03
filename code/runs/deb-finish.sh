#!/bin/bash
# Finish the 44 new train tasks to 65 attempts each on both servers (8006 GPU1, 8007 GPU2), after the GPU2 server
# crash at 23:42 (its errored rollouts were deleted). Rounds repeat (deleting any errored rollouts) until every task
# has 65 error-free attempts; then writes runs/qwen35-train-deb.DONE, which the HF upload and v6 pipelines wait for.
cd /scratch/09059/xliaoyi/post-train-llms
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
for port in 8006 8007; do until curl -sf localhost:$port/v1/models >/dev/null; do sleep 20; done; done
for round in 1 2 3; do
  python3 - $round <<'PY' || break  # exits 1 when nothing is left
import json, glob, os, shutil, sys
tasks = json.load(open("data/new_train_tasks_deb.json"))
for f in glob.glob("runs/qwen35-train/deb/*/*/trajectory.json"):
    if "error" in json.load(open(f))["result"]:
        shutil.rmtree(os.path.dirname(f))
left = {t["task"]: 65 - len(glob.glob(f"runs/qwen35-train/deb/{t['task']}/*/trajectory.json")) for t in tasks}
if not sum(left.values()):
    sys.exit(1)
halves, load = [[], []], [0, 0]
for t in sorted((t for t in tasks if left[t["task"]]), key=lambda t: -left[t["task"]]):  # greedy balance
    i = load.index(min(load)); halves[i].append(t); load[i] += left[t["task"]]
for i, h in enumerate(halves):
    json.dump(h, open(f"data/new_train_tasks_deb_fin{i + 1}.json", "w"), indent=1)
print(f"round {sys.argv[1]}: remaining per server {load}", flush=True)
PY
  cd harness
  for i in 1 2; do
    [ "$(cat ../data/new_train_tasks_deb_fin$i.json)" = "[]" ] && continue
    ../.venv/bin/python agent.py --base-url http://localhost:800$((5 + i))/v1 --model student --tasks-file ../data/new_train_tasks_deb_fin$i.json \
      --attempts 65 --max-turns 100 -j 30 --out ../runs/qwen35-train > ../runs/qwen35-train-deb-fin$i-r$round.log 2>&1 &
  done
  wait; cd ..
  tail -qn1 runs/qwen35-train-deb-fin?-r$round.log
done
python3 - <<'PY' && touch runs/qwen35-train-deb.DONE
import json, glob, sys
tasks = json.load(open("data/new_train_tasks_deb.json"))
res = [json.load(open(f))["result"] for f in glob.glob("runs/qwen35-train/deb/*/*/trajectory.json")]
n = {t["task"]: sum(r["task"] == t["task"] for r in res) for t in tasks}
err = sum("error" in r for r in res)
print(f"{len(res)} rollouts, {sum(r.get('reward') == 1 for r in res)} passed, {err} errors, "
      f"tasks short of 65: {[k for k, v in n.items() if v < 65]}")
sys.exit(err or any(v < 65 for v in n.values()))
PY
date; ls runs/qwen35-train-deb.DONE
