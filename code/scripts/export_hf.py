"""Export rollouts as a Hugging Face SFT dataset (tmax-sft layout: messages, tools, source, metadata).

  python scripts/export_hf.py --runs runs/qwen35-train --name qwen3.5-35b-a3b_self --model Qwen/Qwen3.5-35B-A3B \
      --out hf/dbt-agent-sft [--repo user/dbt-agent-sft]

Writes subsets <name>_all and <name>_only_success (reward == 1) as parquet under --out/data/, and uploads --out
to the dataset repo when --repo is given. Twilio account SIDs in tool output are redacted.
"""
import argparse
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SID = re.compile(r"AC[0-9a-f]{32}")


def row(path, name, model):
    d = json.loads(SID.sub("AC_REDACTED_SID", path.read_text()))
    r = d["result"]
    msgs = [{"role": m["role"], "content": m.get("content") or "", "tool_calls": m.get("tool_calls"),
             "tool_call_id": m.get("tool_call_id")} for m in d["messages"]]
    return {"messages": msgs, "tools": json.dumps(d["tools"]), "source": f"{name}/{r['source']}/{r['task']}",
            "metadata": {"benchmark": r["source"], "task": r["task"], "attempt": r["attempt"],
                         "reward": float(r.get("reward") or 0), "success": r.get("reward") == 1,
                         "stop": r.get("stop"), "num_turns": r.get("turns"), "seconds": r.get("sec"),
                         "model": model, "enable_thinking": False, "max_turns": 100,
                         "error": r.get("error")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="subset prefix")
    ap.add_argument("--model", required=True, help="model that generated the rollouts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo")
    args = ap.parse_args()
    rows = [row(p, args.name, args.model) for d in args.runs for p in sorted(Path(d).rglob("trajectory.json"))]
    out = Path(args.out)
    for subset, keep in [("all", rows), ("only_success", [r for r in rows if r["metadata"]["success"]])]:
        f = out / "data" / f"{args.name}_{subset}" / "train.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(keep), f)
        print(f"{args.name}_{subset}: {len(keep)} rows, {len({r['source'] for r in keep})} tasks -> {f}")
    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(out),
                          commit_message=f"Add {args.name} subsets")


if __name__ == "__main__":
    main()
