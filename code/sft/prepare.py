"""Turn verified agent trajectories into SFT examples with loss on assistant turns only.

  python sft/prepare.py --runs runs/teacher --out data/sft/train.jsonl

Reads every runs/**/trajectory.json with result.reward == 1, renders it with the student's
chat template (non-thinking), and writes {"input_ids", "labels"} per line; labels are -100
outside assistant turns. Assistant spans are exact: the template is prefix-consistent, so the
span of turn i is render(messages[:i+1]) minus render(messages[:i], add_generation_prompt).
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def to_template_messages(messages, drop_call_text=False):
    """Tool-call arguments are JSON strings in the OpenAI format; the template wants dicts.
    drop_call_text empties the text of assistant turns with tool calls (Gemma 4 drops it from history)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            if drop_call_text:
                m["content"] = ""
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        m.pop("student", None)
        out.append(m)
    return out


def render(tok, messages, tools, gen=False):
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=gen,
                                   enable_thinking=False)


def gen_prompt_fix(tok, messages, tools):
    """Some templates (Gemma 4) add text to the generation prompt that history turns lack, e.g. an empty
    thought block after "<|turn>model\\n". Insert it at every assistant turn so training matches inference."""
    i = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    base = render(tok, messages[:i], tools)
    gen = render(tok, messages[:i], tools, gen=True)[len(base):]
    turn = render(tok, messages[:i + 1], tools)[len(base):]
    n = next((k for k in range(min(len(gen), len(turn))) if gen[k] != turn[k]), min(len(gen), len(turn)))
    n = gen.rfind("\n", 0, n) + 1  # back off to a line end so the header never splits a special token
    header, extra = gen[:n], gen[n:]
    if not extra:
        return lambda s: s
    return lambda s: s.replace(header + extra, header).replace(header, header + extra)


def encode(tok, messages, tools, drop_call_text=False):
    """Assistant turns marked {"student": true} (DAgger runs: the student's own turns) get no loss."""
    student = [m.get("student", False) for m in messages]
    messages = to_template_messages(messages, drop_call_text)
    norm = gen_prompt_fix(tok, messages, tools)
    full = norm(render(tok, messages, tools))
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or student[i]:
            continue
        prefix = norm(render(tok, messages[:i], tools, gen=True))
        upto = norm(render(tok, messages[:i + 1], tools))
        assert full.startswith(prefix) and full.startswith(upto), f"template not prefix-consistent at turn {i}"
        spans.append((len(prefix), len(upto)))
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    labels = [tid if any(s <= a < e for s, e in spans) else -100
              for tid, (a, _) in zip(enc["input_ids"], enc["offset_mapping"])]
    return enc["input_ids"], labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories to scan for trajectory.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-2B"))
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--include-failed", action="store_true", help="for pipeline testing only")
    ap.add_argument("--drop-call-text", action="store_true", help="empty assistant text next to tool calls (Gemma 4)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, too_long, lens, trained = 0, 0, [], 0
    with open(out, "w") as f:
        for run_dir in args.runs:
            for p in sorted(Path(run_dir).rglob("trajectory.json")):
                d = json.loads(p.read_text())
                if d["result"].get("reward") != 1 and not args.include_failed:
                    continue
                ids, labels = encode(tok, d["messages"], d["tools"], args.drop_call_text)
                if len(ids) > args.max_len:
                    too_long += 1
                    continue
                f.write(json.dumps({"source": str(p.resolve().relative_to(ROOT)), "input_ids": ids, "labels": labels}) + "\n")
                kept += 1
                lens.append(len(ids))
                trained += sum(l != -100 for l in labels)
    lens.sort()
    print(f"examples={kept} dropped_too_long={too_long} tokens: median={lens[len(lens) // 2] if lens else 0} "
          f"max={lens[-1] if lens else 0} total={sum(lens)} trained_tokens={trained}")


if __name__ == "__main__":
    main()
