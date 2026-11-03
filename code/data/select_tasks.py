"""Write data/train_tasks.json: training tasks (Spider2-DBT + analytics subset of ADE-bench).

Spider2-DBT: all tasks (every task is "complete/build dbt models for an analytical request").
ADE-bench: hand-labelled below; keep model creation, metric/column additions and grain fixes,
drop pure debugging, refactoring, config/package and snapshot tasks.
Tasks whose oracle check (runs/oracle/results.jsonl) is not ok are dropped.
Decontamination vs data-eng-bench: 0 shared instruction 8-grams; data-eng-bench uses its own
retail DB (only generic table names dim_customer/dim_product overlap with spider2 retail001).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from env import ROOT, list_tasks  # noqa: E402

ADE_KEEP = {
    "airbnb005": "create NPS aggregate tables",
    "airbnb007": "create models described in schema.yml",
    "ana-eng003": "create dim_customer with primary key",
    "ana-eng004": "create obt joining product details to inventory",
    "ana-eng005": "fix model grain to one row per inventory_id",
    "ana-eng006": "create dim, fact and obt models",
    "f1002": "create four stats models from yml",
    "f1003": "fix/create top-N analytical models",
    "f1003-hard": "fix/create top-N analytical models",
    "f1009": "create driver current-age analysis model",
    "f1010": "create lap-time trend analysis model",
    "f1010-medium": "create lap-time trend analysis model",
    "f1011": "correct an analyst's lap-time analysis model",
    "helixops-saas001": "add billing_country to dim_accounts",
    "helixops-saas002": "add owning team to account 360",
    "helixops-saas003": "filter sandbox usage from daily usage metric",
    "helixops-saas004": "derive department column in roster",
    "helixops-saas004-no_hint": "derive department column in roster",
    "helixops-saas006": "add net_mrr metric to account 360",
    "helixops-saas007": "add geo_segment through marts",
    "helixops-saas007-no_location_hint": "add geo_segment through marts",
    "helixops-saas010": "filter archived workspaces downstream of staging",
    "helixops-saas016": "update SLA metric logic for enterprise accounts",
    "helixops-saas017": "add department to workspace roster",
    "helixops-saas018": "add net_mrr metric to account 360",
    "intercom001": "create conversation threads aggregate",
    "intercom002": "create threads and conversation metrics models",
    "intercom003": "create conversation metrics aggregate",
}


def main():
    oracle = {}
    for line in open(ROOT / "runs/oracle/results.jsonl"):
        r = json.loads(line)
        oracle[(r["source"], r["task"])] = r["ok"]
    out, dropped = [], []
    for t in list_tasks("spider2"):
        (out if oracle.get(("spider2", t)) else dropped).append({"source": "spider2", "task": t, "reason": "spider2-dbt analytics"})
    for short, reason in ADE_KEEP.items():
        t = f"ade-bench-{short}"
        (out if oracle.get(("ade", t)) else dropped).append({"source": "ade", "task": t, "reason": reason})
    (ROOT / "data/train_tasks.json").write_text(json.dumps(out, indent=1))
    print(f"kept {len(out)} (spider2 {sum(r['source'] == 'spider2' for r in out)}, ade {sum(r['source'] == 'ade' for r in out)})")
    print("dropped (oracle not ok):", [r["task"] for r in dropped])


if __name__ == "__main__":
    main()
