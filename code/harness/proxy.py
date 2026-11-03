"""Logging proxy between an agent CLI and vLLM: same sampling for every harness, no web tools, exact I/O logs.

  python harness/proxy.py --port 8100 --backend http://localhost:8008 http://localhost:8009 [--log-dir runs/proxy-logs] \
      [--thinking off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5]

Agents use base URL http://localhost:8100/r/<run_id>/v1. Every /chat/completions request is forwarded
(non-streaming) to the least-busy backend after overriding sampling (temperature/top_p/top_k/presence_penalty),
setting Qwen's enable_thinking (--thinking on|off), dropping tools whose name contains "web", and capping max_tokens.
With --log-dir, request and response are appended to <log-dir>/<run_id>.jsonl, the record of what the model saw and
produced (used for SFT). Streaming clients get the full response
replayed as SSE chunks, with keep-alive comments while the backend is generating.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from aiohttp import web

DROP = ("presence_penalty", "frequency_penalty", "repetition_penalty", "max_completion_tokens", "stream_options")


def rewrite(body, args):
    body = dict(body)
    for k in DROP:
        body.pop(k, None)
    body["temperature"], body["top_p"], body["top_k"] = args.temperature, args.top_p, args.top_k
    body["presence_penalty"] = args.presence_penalty
    body["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}),
                                    "enable_thinking": args.thinking == "on"}
    body["max_tokens"] = min(body.get("max_tokens") or args.max_tokens, args.max_tokens)
    if body.get("tools"):
        body["tools"] = [t for t in body["tools"] if "web" not in t.get("function", {}).get("name", "").lower()]
        if not body["tools"]:
            body.pop("tools")
            body.pop("tool_choice", None)
    body["stream"] = False
    return body


def sse_chunks(resp, include_usage):
    base = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"], "model": resp["model"]}
    choice = resp["choices"][0]
    msg = choice["message"]
    delta = {"role": "assistant", "content": msg.get("content") or ""}
    for k in ("reasoning", "reasoning_content"):
        if msg.get(k):
            delta[k] = msg[k]
    yield {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    if msg.get("tool_calls"):
        calls = [{"index": i, "id": c["id"], "type": "function", "function": c["function"]}
                 for i, c in enumerate(msg["tool_calls"])]
        yield {**base, "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}]}
    if include_usage:
        yield {**base, "choices": [], "usage": resp.get("usage")}


class Proxy:
    def __init__(self, args):
        self.args = args
        self.inflight = {b: 0 for b in args.backend}
        self.log_dir = Path(args.log_dir) if args.log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    async def start(self, app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=3600))

    async def call(self, body):
        backend = min(self.inflight, key=self.inflight.get)
        self.inflight[backend] += 1
        try:
            async with self.session.post(f"{backend}/v1/chat/completions", json=body) as r:
                return r.status, await r.text(), backend
        finally:
            self.inflight[backend] -= 1

    async def completions(self, request):
        run = request.match_info["run"]
        orig = await request.json()
        body = rewrite(orig, self.args)
        t0 = time.time()
        task = asyncio.ensure_future(self.call(body))
        done, _ = await asyncio.wait({task}, timeout=5)
        stream = None
        if not done and orig.get("stream"):  # long generation: open the stream now and keep it alive
            stream = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await stream.prepare(request)
            while not task.done():
                await stream.write(b": keep-alive\n\n")
                await asyncio.wait({task}, timeout=15)
        try:
            status, text, backend = await task
        except (aiohttp.ClientError, OSError) as e:  # backend down: tell the client and log it as an infra error
            status, text, backend = 502, json.dumps({"error": {"message": f"backend unreachable: {e!r}"}}), None
        if status == 400 and "maximum context length" in text:  # max_tokens too large for the remaining context
            body.pop("max_tokens")
            status, text, backend = await self.call(body)
        try:
            resp = json.loads(text)
        except json.JSONDecodeError:
            resp = {"error": {"message": text}}
        if self.log_dir:
            with open(self.log_dir / f"{run}.jsonl", "a") as f:
                f.write(json.dumps({"t": t0, "secs": round(time.time() - t0, 2), "backend": backend,
                                    "status": status, "request": body, "response": resp}) + "\n")
        if not orig.get("stream"):
            return web.json_response(resp, status=status)
        if stream is None:
            if status != 200:
                return web.json_response(resp, status=status)
            stream = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await stream.prepare(request)
        if status != 200:
            await stream.write(f"data: {json.dumps(resp)}\n\n".encode())
        else:
            include_usage = (orig.get("stream_options") or {}).get("include_usage", False)
            for chunk in sse_chunks(resp, include_usage):
                await stream.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await stream.write(b"data: [DONE]\n\n")
        await stream.write_eof()
        return stream

    async def passthrough(self, request):
        async with self.session.request(request.method, f"{self.args.backend[0]}/v1/{request.match_info['tail']}",
                                        data=await request.read()) as r:
            return web.Response(body=await r.read(), status=r.status, content_type=r.content_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--backend", nargs="+", required=True, help="vLLM servers, e.g. http://localhost:8008")
    ap.add_argument("--log-dir", help="log every request/response here (default: no logging)")
    ap.add_argument("--thinking", choices=["on", "off"], default="on", help="Qwen chat template enable_thinking")
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=32768)
    args = ap.parse_args()
    proxy = Proxy(args)
    app = web.Application(client_max_size=256 * 1024 ** 2)
    app.on_startup.append(proxy.start)
    app.router.add_post("/r/{run}/v1/chat/completions", proxy.completions)
    app.router.add_route("*", "/r/{run}/v1/{tail:.*}", proxy.passthrough)
    web.run_app(app, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
