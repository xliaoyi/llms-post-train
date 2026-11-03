"""DAgger-style rollouts: the student acts until it slips, then Claude takes over from that exact state.

  python harness/dagger.py --base-url http://localhost:8000/v1 --tasks-file data/train_tasks.json \
      --attempts 2 -j 8 --budget 100 --out runs/dagger

The student hands over when it is about to repeat a tool call it already made, when it stops calling tools
(Claude then checks and finishes the work), or after --student-turns turns; the triggering turn is dropped.
Student turns are saved with {"student": true}, so sft/prepare.py trains only on Claude's turns.
"""
import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from openai import OpenAI

from agent import CLAUDE_PRICE, LEDGER, PRICE, SPEND, SYSTEM, TOOLS, clip, cost, run_tool, solve_claude
from env import Env, Task


def student_until_slip(client, env, instruction, max_turns, temperature):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]
    seen = set()
    for _ in range(max_turns):
        resp = client.chat.completions.create(model="student", messages=messages, tools=TOOLS,
                                              temperature=temperature, max_completion_tokens=16384,
                                              extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return messages, "stopped"
        calls = [(c.function.name, c.function.arguments) for c in msg.tool_calls]
        if any(c in seen for c in calls):
            return messages, "repeat"
        seen.update(calls)
        messages.append({"role": "assistant", "content": msg.content or "", "student": True,
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in msg.tool_calls]})
        for c in msg.tool_calls:
            try:
                out = run_tool(env, c.function.name, json.loads(c.function.arguments))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                out = f"[error] bad tool arguments: {e}"
            messages.append({"role": "tool", "tool_call_id": c.id, "content": clip(out)})
    return messages, "student_turns"


def run_one(args, source, task_id, attempt):
    out = Path(args.out) / source / task_id / str(attempt)
    if (out / "trajectory.json").exists():
        return json.loads((out / "trajectory.json").read_text())["result"]
    t0 = time.time()
    res = {"source": source, "task": task_id, "attempt": attempt, "model": args.model}
    usage = {"input": 0, "cached": 0, "cache_write": 0, "output": 0}
    messages, env = [], None
    try:
        task = Task(source, task_id)
        env = Env(task, out)
        student = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
        messages, trigger = student_until_slip(student, env, task.instruction, args.student_turns, args.temperature)
        res.update(trigger=trigger, student_turns=sum(m["role"] == "assistant" for m in messages))
        if SPEND["usd"] >= SPEND["budget"]:
            res.update(stop="budget", reward=0)
        else:
            claude = anthropic.Anthropic(timeout=600, max_retries=8)
            messages, stop = solve_claude(claude, args.model, env, task.instruction, args.max_turns, args.effort,
                                          usage, messages=messages)
            res.update(stop=stop, turns=sum(m["role"] == "assistant" for m in messages), reward=env.verify())
    except Exception:
        res.update(error=traceback.format_exc()[-2000:], reward=0)
    finally:
        if env:
            env.close()
    res["sec"] = round(time.time() - t0)
    res["usage"] = usage
    res["usd"] = round(cost(usage), 4)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectory.json").write_text(json.dumps({"result": res, "tools": TOOLS, "messages": messages}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server for the student")
    ap.add_argument("--model", default="claude-opus-5-5", choices=list(CLAUDE_PRICE))
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--student-turns", type=int, default=30)
    ap.add_argument("--max-turns", type=int, default=40, help="Claude turns after the takeover")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--budget", type=float, required=True, help="cap on total Claude spend in runs/claude-spend.jsonl")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    PRICE.update(zip(["input", "cached", "output"], CLAUDE_PRICE[args.model]))
    SPEND["budget"] = args.budget
    if LEDGER.exists():
        SPEND["usd"] = sum(json.loads(line)["usd"] for line in LEDGER.open())
    print(f"Claude spend so far: ${SPEND['usd']:.2f}", flush=True)
    jobs = [(t["source"], t["task"], a) for t in json.load(open(args.tasks_file)) for a in range(args.attempts)]
    print(f"{len(jobs)} runs", flush=True)
    done = ok = 0
    with ThreadPoolExecutor(args.j) as pool:
        for fut in as_completed([pool.submit(run_one, args, *j) for j in jobs]):
            r = fut.result()
            done += 1
            ok += r.get("reward", 0)
            print(f"[{done}/{len(jobs)} pass={ok}] {r['source']}/{r['task']}#{r['attempt']} reward={r.get('reward')} "
                  f"trigger={r.get('trigger')}@{r.get('student_turns')} stop={r.get('stop')} turns={r.get('turns')} "
                  f"{r['sec']}s ${r['usd']} total=${SPEND['usd']:.2f} {'ERROR' if 'error' in r else ''}", flush=True)
    print(f"TOTAL {ok}/{len(jobs)} = {100 * ok / len(jobs):.1f}%  spent=${SPEND['usd']:.2f}")


if __name__ == "__main__":
    main()
