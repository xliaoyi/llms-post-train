"""Evaluate each step checkpoint of a running sft/train.py on train and test tasks; early-stop training; log to wandb.

  python sft/eval_loop.py --run checkpoints/sft-v6 --gpu <uuid> [<uuid> ...] --port 8008 [8009 ...] --train-pid <pid> \
      --set train:data/train_tasks_v2.json:1:0.21 --set test:data/test_tasks_v2.json:4:0.15

Each --set is name:tasks_file:attempts:baseline (baseline = base-model success rate, plotted as step 0); the
first set drives early stopping. Repeating a name with disjoint task files splits that set over more servers. For every <run>/step<N> with READY (written by train.py --save-every), serves
it with vLLM (one server per --gpu; set i uses server i % n), runs the agent on every set concurrently, and writes <run>/step<N>/eval.json, <run>/success.jsonl
and the plot <run>/success_vs_step.png. Only the best checkpoint of each set keeps its weights. Training is
stopped once --patience evaluations in a row fail to beat the best success rate of the first set. With
--after-train the checkpoints are evaluated in order once training exits, stopping after the same --patience rule.
train_log.jsonl and the evals are mirrored to wandb (this is the only process writing to the wandb run).
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ROOT = Path(__file__).resolve().parents[1]
SERVE_ARGS = '--quantization fp8 --max-num-seqs 64 --limit-mm-per-prompt {"image":0,"video":0}'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def serve(ckpt, gpu, port, log):
    env = {**os.environ, "MEM": "0.85", "VLLM_ARGS": SERVE_ARGS}
    proc = subprocess.Popen(["bash", "scripts/serve.sh", gpu, str(port), str(ckpt.resolve())], cwd=ROOT, env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            return proc
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited, see {log}")
            time.sleep(10)
    raise RuntimeError(f"vLLM did not start, see {log}")


def stop(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=300)


def start_rollouts(port, out, s, args):
    return subprocess.Popen([str(ROOT / ".venv/bin/python"), "agent.py", "--base-url", f"http://localhost:{port}/v1",
                             "--model", "student", "--tasks-file", str(ROOT / s["tasks_file"]),
                             "--attempts", str(s["attempts"]), "--max-turns", "100", "-j", str(args.j),
                             "--out", str(out.resolve())],
                            cwd=ROOT / "harness", stdout=open(f"{out}.{Path(s['tasks_file']).stem}.log", "w"), stderr=subprocess.STDOUT)


def infra_error(r):
    return "error" in r and "maximum context length" not in r["error"]


def score(out):
    res = [json.loads(p.read_text())["result"] for p in out.glob("*/*/*/trajectory.json")]
    passed = sum(r.get("reward") == 1 for r in res)
    solved = len({(r["source"], r["task"]) for r in res if r.get("reward") == 1})
    errors = sum("error" in r for r in res)
    return {"runs": len(res), "passed": passed, "success_rate": passed / len(res), "tasks_solved": solved,
            "errors": errors}


def plot(run, records, sets):
    log = [json.loads(line) for line in open(run / "train_log.jsonl")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for s in sets:
        steps = [0] + [r["step"] for r in records]
        rates = [100 * s["baseline"]] + [100 * r[s["name"]]["success_rate"] for r in records]
        line, = a.plot(steps, rates, "o-", label=f"{s['name']} ({s['attempts']} attempt(s)/task)")
        best = max(range(len(rates)), key=lambda i: (rates[i], -i))
        a.plot(steps[best], rates[best], "*", color=line.get_color(), ms=14,
               label=f"best {s['name']}: step {steps[best]}, {rates[best]:.1f}%")
    a.set_ylabel("success rate (%)")
    a.set_title(f"{run.name}: task success rate vs training step")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    b.plot([r["step"] for r in log], [r["loss"] for r in log], lw=0.8)
    b.set_xlabel("optimizer step")
    b.set_ylabel("training loss")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "success_vs_step.png", dpi=150)
    plt.close(fig)


def sync_train_log(run, wb, logged):
    """Forward new train_log.jsonl lines to wandb; returns the last step logged."""
    if not (run / "train_log.jsonl").exists():
        return logged
    for line in open(run / "train_log.jsonl"):
        r = json.loads(line)
        if r["step"] > logged:
            wb.log({"train/step": r["step"], **{f"train/{k}": v for k, v in r.items() if k != "step"}})
            logged = r["step"]
    return logged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train.py --out directory")
    ap.add_argument("--gpu", nargs="+", required=True, help="one vLLM server per GPU")
    ap.add_argument("--port", type=int, nargs="+", default=[8008, 8009])
    ap.add_argument("--train-pid", type=int, required=True)
    ap.add_argument("--set", action="append", required=True, help="name:tasks_file:attempts:baseline")
    ap.add_argument("-j", type=int, default=30, help="concurrent rollouts per set")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--after-train", action="store_true",
                    help="training uses the eval GPUs: only sync the loss until it exits, then evaluate the checkpoints")
    ap.add_argument("--wandb-entity", default="llms-post-train")
    ap.add_argument("--wandb-project", default="llms-post-train")
    args = ap.parse_args()
    parts = [dict(zip(["name", "tasks_file", "attempts", "baseline"], s.split(":"))) for s in args.set]
    for s in parts:
        s["attempts"], s["baseline"] = int(s["attempts"]), float(s["baseline"])
    sets = list({s["name"]: s for s in reversed(parts)}.values())[::-1]  # repeated names = one set split over servers
    key = sets[0]["name"]
    run = ROOT / args.run
    rollout_dir = ROOT / "runs" / f"{run.name}-eval"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    id_file = run / "wandb_id"
    if not id_file.exists():
        id_file.write_text(uuid.uuid4().hex[:8])
    config = {**json.loads((run / "args.json").read_text()), "eval_sets": sets}
    wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run.name, id=id_file.read_text(),
                    resume="allow", config=config, dir=str(run))
    wb.define_metric("train/step")
    wb.define_metric("train/*", step_metric="train/step")
    wb.define_metric("eval/step")
    wb.define_metric("eval/*", step_metric="eval/step")
    logged = wb.summary.get("train/step", 0)
    records = [json.loads(line) for line in open(run / "success.jsonl")] if (run / "success.jsonl").exists() else []
    if not records and wb.summary.get("eval/step") is None:
        wb.log({"eval/step": 0, **{f"eval/{s['name']}/success_rate": s["baseline"] for s in sets}})
    best, best_ckpt = {s["name"]: s["baseline"] for s in sets}, {s["name"]: None for s in sets}
    for r in records:  # resuming
        for n in best:
            if r[n]["success_rate"] > best[n]:
                best[n], best_ckpt[n] = r[n]["success_rate"], run / f"step{r['step']}"
    since_best = 0
    while True:
        logged = sync_train_log(run, wb, logged)
        pending = sorted((p for p in run.glob("step*") if (p / "READY").exists() and not (p / "eval.json").exists()),
                         key=lambda p: int(p.name[4:]))
        if not pending or (args.after_train and alive(args.train_pid)):  # after-train: training holds the GPUs
            if not pending and not alive(args.train_pid):
                break
            time.sleep(60)
            continue
        ckpt = pending[0]
        step = int(ckpt.name[4:])
        outs = {s["name"]: rollout_dir / s["name"] / ckpt.name for s in sets}
        for o in outs.values():
            o.parent.mkdir(exist_ok=True)
        for attempt in range(3):  # a vLLM crash turns the remaining rollouts into errors: drop them and resume
            servers = []  # one server per GPU, all serving this checkpoint; set i uses server i % len(gpus)
            try:
                for g, port in zip(args.gpu, args.port):
                    servers.append(serve(ckpt, g, port, ROOT / "logs" / f"vllm-{run.name}-{ckpt.name}-{port}-{attempt}.log"))
                procs = [start_rollouts(args.port[i % len(args.port)], outs[s["name"]], s, args)
                         for i, s in enumerate(parts)]
                while any(p.poll() is None for p in procs):  # keep the loss curve live while evaluating
                    logged = sync_train_log(run, wb, logged)
                    time.sleep(60)
                if any(p.returncode for p in procs):
                    raise RuntimeError(f"agent.py failed, see {rollout_dir}/*/{ckpt.name}.log")
            finally:
                for proc in servers:
                    stop(proc)
            # context overflow is the model's own failure (reward 0); anything else is infrastructure: rerun
            errored = [p.parent for o in outs.values() for p in o.glob("*/*/*/trajectory.json")
                       if infra_error(json.loads(p.read_text())["result"])]
            if not errored:
                break
            print(f"step {step}: {len(errored)} errored rollouts, rerunning them", flush=True)
            for d in errored:
                shutil.rmtree(d)
        rec = {"step": step, **{name: score(o) for name, o in outs.items()}}
        records.append(rec)
        with open(run / "success.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        plot(run, records, sets)
        wb.log({"eval/step": step, **{f"eval/{s['name']}/{k}": v for s in sets for k, v in rec[s["name"]].items()},
                "eval/plot": wandb.Image(str(run / "success_vs_step.png"))})
        improved = {}
        for s in sets:
            n = s["name"]
            improved[n] = rec[n]["success_rate"] > best[n]
            if improved[n]:
                best[n], prev = rec[n]["success_rate"], best_ckpt[n]
                best_ckpt[n] = ckpt
                if prev and prev not in best_ckpt.values():
                    for w in prev.glob("*.safetensors"):
                        w.unlink()
        if ckpt not in best_ckpt.values():
            for w in ckpt.glob("*.safetensors"):
                w.unlink()
        # eval.json last: train.py --wait-eval resumes once it exists, and weights must be gone by then (disk)
        (ckpt / "eval.json").write_text(json.dumps(rec, indent=1))
        since_best = 0 if improved[key] else since_best + 1
        print(f"step {step}: " + ", ".join(f"{n} {100 * rec[n]['success_rate']:.1f}% "
                                          f"({rec[n]['passed']}/{rec[n]['runs']}, best {100 * best[n]:.1f}%)"
                                          for n in best) + f"; {since_best} evals since best {key}", flush=True)
        if since_best >= args.patience:
            print(f"no {key} improvement in {args.patience} evaluations: stopping training (pid {args.train_pid})",
                  flush=True)
            if alive(args.train_pid):
                os.kill(args.train_pid, signal.SIGTERM)
            break
    sync_train_log(run, wb, logged)
    best_txt = {n: f"{c.name if c else 'base model'} {100 * best[n]:.1f}%" for n, c in best_ckpt.items()}
    (run / "BEST").write_text("".join(f"{n}: {t}\n" for n, t in best_txt.items()))
    for n, t in best_txt.items():
        wb.summary[f"best/{n}"] = t
    wb.finish()


if __name__ == "__main__":
    main()
