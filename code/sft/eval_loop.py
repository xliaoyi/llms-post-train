"""Evaluate each step checkpoint of a running sft/train.py on the train tasks; early-stop training.

  python sft/eval_loop.py --run checkpoints/sft-v5 --gpu <uuid> --port 8008 --train-pid <pid>

For every <run>/step<N> with READY (written by train.py --save-every), serves it with vLLM, runs the
agent on data/train_tasks.json, and writes <run>/step<N>/eval.json, <run>/success.jsonl and the plot
<run>/success_vs_step.png. Only the best checkpoint keeps its weights. Training is stopped once
--patience evaluations in a row fail to beat the best success rate.
"""
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def rollouts(port, out, args):
    subprocess.run([str(ROOT / ".venv/bin/python"), "agent.py", "--base-url", f"http://localhost:{port}/v1",
                    "--model", "student", "--tasks-file", str(ROOT / args.tasks_file), "--attempts", str(args.attempts),
                    "--max-turns", "100", "-j", str(args.j), "--out", str(out.resolve())],
                   cwd=ROOT / "harness", stdout=open(f"{out}.log", "w"), stderr=subprocess.STDOUT, check=True)
    res = [json.loads(p.read_text())["result"] for p in out.glob("*/*/*/trajectory.json")]
    passed = sum(r.get("reward") == 1 for r in res)
    solved = len({(r["source"], r["task"]) for r in res if r.get("reward") == 1})
    errors = sum("error" in r for r in res)
    return {"runs": len(res), "passed": passed, "success_rate": passed / len(res), "tasks_solved": solved,
            "errors": errors}


def plot(run, records, baseline):
    steps = [0] + [r["step"] for r in records]
    rates = [100 * baseline] + [100 * r["success_rate"] for r in records]
    log = [json.loads(line) for line in open(run / "train_log.jsonl")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    a.plot(steps, rates, "o-")
    best = max(range(len(rates)), key=lambda i: (rates[i], -i))
    a.plot(steps[best], rates[best], "r*", ms=14, label=f"best: step {steps[best]}, {rates[best]:.1f}%")
    a.set_ylabel("train-task success rate (%)")
    a.set_title(f"{run.name}: success rate on train tasks vs training step")
    a.legend()
    a.grid(alpha=0.3)
    b.plot([r["step"] for r in log], [r["loss"] for r in log], lw=0.8)
    b.set_xlabel("optimizer step")
    b.set_ylabel("training loss")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "success_vs_step.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train.py --out directory")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--train-pid", type=int, required=True)
    ap.add_argument("--tasks-file", default="data/train_tasks.json")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("-j", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--baseline", type=float, required=True, help="success rate of the base model (step 0)")
    args = ap.parse_args()
    run = ROOT / args.run
    rollout_dir = ROOT / "runs" / f"{run.name}-train-eval"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in open(run / "success.jsonl")] if (run / "success.jsonl").exists() else []
    best = max([args.baseline] + [r["success_rate"] for r in records])
    best_ckpt, since_best = None, 0
    while True:
        pending = sorted((p for p in run.glob("step*") if (p / "READY").exists() and not (p / "eval.json").exists()),
                         key=lambda p: int(p.name[4:]))
        if not pending:
            if not alive(args.train_pid):
                break
            time.sleep(60)
            continue
        ckpt = pending[0]
        step = int(ckpt.name[4:])
        proc = serve(ckpt, args.gpu, args.port, ROOT / "logs" / f"vllm-{run.name}-{ckpt.name}.log")
        try:
            rec = {"step": step, **rollouts(args.port, rollout_dir / ckpt.name, args)}
        finally:
            stop(proc)
        (ckpt / "eval.json").write_text(json.dumps(rec, indent=1))
        records.append(rec)
        with open(run / "success.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        plot(run, records, args.baseline)
        if rec["success_rate"] > best:
            best, since_best = rec["success_rate"], 0
            if best_ckpt:
                for w in best_ckpt.glob("*.safetensors"):
                    w.unlink()
            best_ckpt = ckpt
        else:
            since_best += 1
            for w in ckpt.glob("*.safetensors"):
                w.unlink()
        print(f"step {step}: {100 * rec['success_rate']:.1f}% ({rec['passed']}/{rec['runs']}), "
              f"best {100 * best:.1f}%, {since_best} evals since best", flush=True)
        if since_best >= args.patience:
            print(f"no improvement in {args.patience} evaluations: stopping training (pid {args.train_pid})", flush=True)
            if alive(args.train_pid):
                os.kill(args.train_pid, signal.SIGTERM)
            break
    (run / "BEST").write_text(f"{best_ckpt.name if best_ckpt else 'base model'} {100 * best:.1f}%\n")


if __name__ == "__main__":
    main()
