"""Minimal tool-calling agent + batch runner. Same scaffold for base student, SFT student and teacher.

  python harness/agent.py --base-url http://localhost:8000/v1 --model student \
      --source deb --category analytics --attempts 3 -j 16 --out runs/eval/base

Each run saves runs/.../<task>/<attempt>/{trajectory.json, verify.log, reward.txt}.
trajectory.json["messages"] is in OpenAI chat format and is used directly as SFT data.
"""
import argparse
import base64
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from openai import OpenAI

from env import Env, Task, list_tasks

SYSTEM = """You are an analytics engineer working inside a DuckDB/dbt project.
Complete the requested analytics-engineering task in the existing project. Inspect the relevant project files and data, implement the requested analytical result, run the necessary dbt commands, inspect execution results, and repair the implementation when necessary. Do not access reference solutions or modify the evaluation mechanism.
Use the tools to act. When the task is complete, reply with a short summary and no tool call."""

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command. Each call starts a fresh shell in the project directory. Returns exit code and output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace one exact occurrence of old_str with new_str in a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"},
                                                        "new_str": {"type": "string"}},
                       "required": ["path", "old_str", "new_str"]}}},
]

MAX_OBS = 8000  # characters of tool output shown to the model

# USD per 1M tokens (gpt-5.6-luna) and a spend counter shared by all runs, so --budget is a hard cap
PRICE = {"input": 0.20, "cached": 0.02, "output": 1.20}
SPEND = {"usd": 0.0, "budget": float("inf"), "lock": threading.Lock()}


# Claude API: USD per 1M tokens (input, cache read, output); cache writes cost 1.25x input.
# Every request's cost is appended to LEDGER so --budget caps total Claude spend across all runs and processes.
CLAUDE_PRICE = {"claude-opus-5-5": (4.00, 0.20, 20.00), "claude-sonnet-5-5": (2.00, 0.20, 10.00)}
LEDGER = Path(__file__).resolve().parent.parent / "runs" / "claude-spend.jsonl"


def cost(usage):
    return ((usage["input"] - usage["cached"] - usage["cache_write"]) * PRICE["input"]
            + usage["cached"] * PRICE["cached"] + usage["cache_write"] * 1.25 * PRICE["input"]
            + usage["output"] * PRICE["output"]) / 1e6


def clip(text):
    if len(text) <= MAX_OBS:
        return text
    half = MAX_OBS // 2
    return f"{text[:half]}\n... [{len(text) - MAX_OBS} characters omitted] ...\n{text[-half:]}"


def b64(s):
    return base64.b64encode(s.encode()).decode()


def run_tool(env, name, args):
    if name == "bash":
        code, out = env.exec(args["command"])
        return f"[exit {code}]\n{out}"
    if name == "read_file":
        code, out = env.exec(f"cat -- {json.dumps(args['path'])}")
        return out if code == 0 else f"[error]\n{out}"
    if name == "write_file":
        p = json.dumps(args["path"])
        code, out = env.exec(f"mkdir -p \"$(dirname {p})\" && echo {b64(args['content'])} | base64 -d > {p}")
        return "ok" if code == 0 else f"[error]\n{out}"
    if name == "edit_file":
        script = ("import sys,base64;p,o,n=[base64.b64decode(a).decode() for a in sys.argv[1:]];"
                  "t=open(p).read();c=t.count(o);"
                  "sys.exit(f'old_str found {c} times, must be exactly 1') if c!=1 else open(p,'w').write(t.replace(o,n));print('ok')")
        code, out = env.exec(f"python3 -c {json.dumps(script)} {b64(args['path'])} {b64(args['old_str'])} {b64(args['new_str'])}")
        return out if code == 0 else f"[error]\n{out}"
    return f"[error] unknown tool {name}"


def solve(client, model, env, instruction, max_turns, temperature, request_kwargs, usage):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]
    for _ in range(max_turns):
        if SPEND["usd"] >= SPEND["budget"]:
            return messages, "budget"
        resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS,
                                              temperature=temperature, max_completion_tokens=16384,
                                              **request_kwargs)
        if resp.usage:
            details = resp.usage.prompt_tokens_details
            step = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens,
                    "cached": (details.cached_tokens or 0) if details else 0, "cache_write": 0}
            for k in usage:
                usage[k] += step[k]
            with SPEND["lock"]:
                SPEND["usd"] += cost(step)
        msg = resp.choices[0].message
        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                   for c in msg.tool_calls]
        messages.append(entry)
        if not msg.tool_calls:
            return messages, "finished"
        for c in msg.tool_calls:
            try:
                out = run_tool(env, c.function.name, json.loads(c.function.arguments))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                out = f"[error] bad tool arguments: {e}"
            messages.append({"role": "tool", "tool_call_id": c.id, "content": clip(out)})
    return messages, "max_turns"


def solve_responses(client, model, env, instruction, max_turns, effort, usage):
    """Same loop via the OpenAI Responses API, which allows reasoning together with function tools.
    Reasoning stays hidden (encrypted, passed back each turn); the saved messages are chat format."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]
    tools = [{"type": "function", **t["function"]} for t in TOOLS]
    items = [{"role": "user", "content": instruction}]
    for _ in range(max_turns):
        if SPEND["usd"] >= SPEND["budget"]:
            return messages, "budget"
        resp = client.responses.create(model=model, instructions=SYSTEM, input=items, tools=tools,
                                       reasoning={"effort": effort}, max_output_tokens=16384, store=False,
                                       include=["reasoning.encrypted_content"])
        step = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                "cached": resp.usage.input_tokens_details.cached_tokens or 0, "cache_write": 0}
        for k in usage:
            usage[k] += step[k]
        with SPEND["lock"]:
            SPEND["usd"] += cost(step)
        items += [o.model_dump(exclude_none=True) for o in resp.output]
        calls = [o for o in resp.output if o.type == "function_call"]
        entry = {"role": "assistant", "content": resp.output_text or ""}
        if calls:
            entry["tool_calls"] = [{"id": c.call_id, "type": "function",
                                    "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
        messages.append(entry)
        if not calls:
            return messages, "finished"
        for c in calls:
            try:
                out = clip(run_tool(env, c.name, json.loads(c.arguments)))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                out = f"[error] bad tool arguments: {e}"
            messages.append({"role": "tool", "tool_call_id": c.call_id, "content": out})
            items.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
    return messages, "max_turns"


def to_anthropic(messages):
    """Chat-format history (system, user, assistant/tool turns) -> Anthropic messages."""
    history = []
    for m in messages[1:]:
        if m["role"] == "user":
            history.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
            for c in m.get("tool_calls", []):
                try:
                    args = json.loads(c["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {"raw": c["function"]["arguments"]}
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": args})
            history.append({"role": "assistant", "content": blocks})
        else:
            result = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if history[-1]["role"] == "user" and isinstance(history[-1]["content"], list):
                history[-1]["content"].append(result)
            else:
                history.append({"role": "user", "content": [result]})
    return history


def solve_claude(client, model, env, instruction, max_turns, effort, usage, messages=None):
    """Same loop via the Anthropic Messages API (adaptive thinking, hidden). The API history keeps the raw
    content blocks (thinking blocks must be passed back unchanged); the saved messages are chat format.
    With messages, Claude continues that chat-format history (e.g. a student's partial run)."""
    messages = messages or [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]
    tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
              "input_schema": t["function"]["parameters"]} for t in TOOLS]
    history = to_anthropic(messages)
    for _ in range(max_turns):
        if SPEND["usd"] >= SPEND["budget"]:
            return messages, "budget"
        resp = client.messages.create(model=model, max_tokens=16000, system=SYSTEM, tools=tools, messages=history,
                                      output_config={"effort": effort}, cache_control={"type": "ephemeral"})
        u = resp.usage
        step = {"input": u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens,
                "cached": u.cache_read_input_tokens, "cache_write": u.cache_creation_input_tokens,
                "output": u.output_tokens}
        for k in usage:
            usage[k] += step[k]
        with SPEND["lock"]:
            SPEND["usd"] += cost(step)
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"t": time.time(), "model": model, "usd": cost(step)}) + "\n")
        if resp.stop_reason in ("refusal", "max_tokens"):
            return messages, resp.stop_reason
        history.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        entry = {"role": "assistant", "content": "".join(b.text for b in resp.content if b.type == "text")}
        if calls:
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": json.dumps(c.input)}} for c in calls]
        messages.append(entry)
        if not calls:
            return messages, "finished"
        results = []
        for c in calls:
            try:
                out = clip(run_tool(env, c.name, c.input))
            except (KeyError, TypeError) as e:
                out = f"[error] bad tool arguments: {e}"
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
            results.append({"type": "tool_result", "tool_use_id": c.id, "content": out})
        history.append({"role": "user", "content": results})
    return messages, "max_turns"


def run_one(args, source, task_id, attempt):
    out = Path(args.out) / source / task_id / str(attempt)
    if (out / "trajectory.json").exists():
        return json.loads((out / "trajectory.json").read_text())["result"]
    t0 = time.time()
    res = {"source": source, "task": task_id, "attempt": attempt}
    usage = {"input": 0, "cached": 0, "cache_write": 0, "output": 0}
    env = None
    try:
        task = Task(source, task_id)
        env = Env(task, out)
        if args.model in CLAUDE_PRICE:
            client = anthropic.Anthropic(timeout=600, max_retries=8)  # key from ANTHROPIC_API_KEY
        elif args.portkey_provider:  # course gateway to OpenAI; function tools need reasoning_effort="none"
            key = Path(args.api_key_file).read_text().strip()
            client = OpenAI(base_url="https://api.portkey.ai/v1", api_key="unused", timeout=600,
                            default_headers={"x-portkey-api-key": key, "x-portkey-provider": args.portkey_provider})
            request_kwargs = {"reasoning_effort": "none"}
        else:
            client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
            request_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": args.thinking}}}
            if args.repetition_penalty:  # vLLM: penalizes tokens already in the prompt or output
                request_kwargs["extra_body"]["repetition_penalty"] = args.repetition_penalty
        if args.model in CLAUDE_PRICE:
            messages, stop = solve_claude(client, args.model, env, task.instruction, args.max_turns, args.effort,
                                          usage)
        elif args.reasoning_effort:
            messages, stop = solve_responses(client, args.model, env, task.instruction, args.max_turns,
                                             args.reasoning_effort, usage)
        else:
            messages, stop = solve(client, args.model, env, task.instruction, args.max_turns, args.temperature,
                                   request_kwargs, usage)
        res.update(stop=stop, turns=sum(m["role"] == "assistant" for m in messages), reward=env.verify())
    except Exception:
        messages = []
        res.update(error=traceback.format_exc()[-2000:], reward=0)
    finally:
        if env:
            env.close()
    res["sec"] = round(time.time() - t0)
    res["usage"] = usage
    res["usd"] = round(cost(usage), 4)
    res["model"] = args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectory.json").write_text(json.dumps({"result": res, "tools": TOOLS, "messages": messages}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", nargs="+", choices=["deb", "spider2", "ade"])
    ap.add_argument("--category")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition-penalty", type=float, help="local vLLM student only")
    ap.add_argument("--thinking", action="store_true", help="enable the student's thinking mode (needs a server with --reasoning-parser)")
    ap.add_argument("--portkey-provider", help="e.g. @30845-fall26-openai; route through the Portkey gateway")
    ap.add_argument("--api-key-file", default=".secrets/portkey_key")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    help="with --portkey-provider: use the Responses API with reasoning")
    ap.add_argument("--budget", type=float, help="stop sending requests once this many USD are spent "
                    "(Claude models: total over all runs recorded in runs/claude-spend.jsonl)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude models: output_config.effort")
    ap.add_argument("--limit", type=int, help="use only the first N tasks of --tasks-file")
    ap.add_argument("--tasks-file", help="JSON list of {source, task} (e.g. data/train_tasks.json)")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.budget:
        SPEND["budget"] = args.budget
    if args.model in CLAUDE_PRICE:
        PRICE.update(zip(["input", "cached", "output"], CLAUDE_PRICE[args.model]))
        if LEDGER.exists():
            SPEND["usd"] = sum(json.loads(line)["usd"] for line in LEDGER.open())
        print(f"Claude spend so far: ${SPEND['usd']:.2f}", flush=True)
    elif not args.portkey_provider:  # local vLLM server: tokens are free
        PRICE.update(input=0, cached=0, output=0)
    if args.tasks_file:
        jobs = [(t["source"], t["task"], a) for t in json.load(open(args.tasks_file))[:args.limit or None]
                for a in range(args.attempts)]
    else:
        jobs = [(s, t, a) for s in args.source for t in (args.tasks or list_tasks(s, args.category))
                for a in range(args.attempts)]
    jobs.sort(key=lambda j: j[2])  # attempt-major, so a --budget stop leaves whole attempts
    print(f"{len(jobs)} runs", flush=True)
    done = ok = 0
    with ThreadPoolExecutor(args.j) as pool:
        for fut in as_completed([pool.submit(run_one, args, *j) for j in jobs]):
            r = fut.result()
            done += 1
            ok += r.get("reward", 0)
            print(f"[{done}/{len(jobs)} pass={ok}] {r['source']}/{r['task']}#{r['attempt']} reward={r.get('reward')} "
                  f"stop={r.get('stop')} turns={r.get('turns')} {r['sec']}s ${r.get('usd', 0)} total=${SPEND['usd']:.2f} "
              f"{'ERROR' if 'error' in r else ''}", flush=True)
    print(f"TOTAL {ok}/{len(jobs)} = {100 * ok / len(jobs):.1f}%  spent=${SPEND['usd']:.2f}")


if __name__ == "__main__":
    main()
