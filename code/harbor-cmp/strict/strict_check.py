"""Strict semantic check, run by the verifier after the task's own test (/tests/strict_check.py /tests/strict.json).

Exits 0 if the trial's output matches the reference output row for row, else 1 (reasons on stdout).
  kind "gold":    every table in /tests/strict_gold.duckdb (the reference solution's output, see build_strict.py) must
                  exist in the trial's database with the same rows: same row count (grain), columns matched by name,
                  numbers within ABS_TOL / REL_TOL. Columns that differ between two reference runs are skipped.
  kind "spider2": Spider2's own check matches each gold column to any output column independently, with each column
                  sorted on its own when row order is ignored, so rows are never compared. Here the matched columns
                  must also line up as rows.
  kind "same":    the task's own grader already compares rows against reference data (ADE); nothing to add.
"""
import itertools
import json
import math
import sys
from datetime import date, datetime
from decimal import Decimal

import duckdb

ABS_TOL, REL_TOL = 0.01, 1e-4


def norm(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, Decimal)):
        v = float(v)
        return None if math.isnan(v) else v
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v if v is None else str(v)


def key(row):  # sort key: numbers rounded so near-equal values sort together
    return tuple((0, "") if v is None else (1, round(v, 2)) if isinstance(v, float) else (2, v) for v in row)


def same_value(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL)
    return a == b


def same_rows(gold, pred, ordered=False):
    """(ok, reason) for two lists of row tuples."""
    if len(gold) != len(pred):
        return False, f"{len(pred)} rows, expected {len(gold)}"
    gold, pred = [tuple(map(norm, r)) for r in gold], [tuple(map(norm, r)) for r in pred]
    if not ordered:
        gold, pred = sorted(gold, key=key), sorted(pred, key=key)
    bad = sum(not all(map(same_value, g, p)) for g, p in zip(gold, pred))
    return (bad == 0), f"{bad} of {len(gold)} rows differ"


def check_gold(cfg):
    con = duckdb.connect(cfg["db"], read_only=True)
    con.execute("ATTACH '/tests/strict_gold.duckdb' AS gold (READ_ONLY)")
    ok = True
    for t in cfg["tables"]:
        cols = t["columns"]
        try:
            have = {r[0].lower(): r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_catalog = current_database() "
                "AND lower(table_schema) = lower(?) AND lower(table_name) = lower(?)", [t["schema"], t["name"]]).fetchall()}
            missing = [c for c in cols if c.lower() not in have]
            if not have or missing:
                print(f"FAIL {t['schema']}.{t['name']}: " + ("table missing" if not have else f"missing columns {missing}"))
                ok = False
                continue
            sel = ", ".join(f'"{have[c.lower()]}"' for c in cols)
            pred = con.execute(f'SELECT {sel} FROM "{t["schema"]}"."{t["name"]}"').fetchall()
            gsel = ", ".join(f'"{c}"' for c in cols)
            gold = con.execute(f'SELECT {gsel} FROM gold."{t["schema"]}"."{t["name"]}"').fetchall()
        except duckdb.Error as e:
            print(f"FAIL {t['schema']}.{t['name']}: {e}")
            ok = False
            continue
        good, why = same_rows(gold, pred)
        print(f"{'ok  ' if good else 'FAIL'} {t['schema']}.{t['name']} ({len(cols)} columns): {why}")
        ok &= good
    return ok


def check_spider2(cfg):
    con = duckdb.connect(cfg["result_db"], read_only=True)
    gcon = duckdb.connect(cfg["gold_db"], read_only=True)
    alias = cfg["result_db"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    try:
        gcon.execute(f"ATTACH '{cfg['result_db']}' AS {alias} (READ_ONLY)")
    except duckdb.Error:
        pass
    ok = True
    for i, table in enumerate(cfg["condition_tabs"]):
        cols = (cfg.get("condition_cols") or [[]] * (i + 1))[i]
        ordered = not (cfg.get("ignore_orders") or [False] * (i + 1))[i]
        try:
            pred = con.execute(f"SELECT * FROM {table}").fetchall()
            gold = gcon.execute(f"SELECT * FROM {table}").fetchall()
        except duckdb.Error as e:
            print(f"FAIL {table}: {e}")
            ok = False
            continue
        gold = [tuple(r[c] for c in cols) for r in gold] if cols else gold
        if len(gold) != len(pred):
            print(f"FAIL {table}: {len(pred)} rows, expected {len(gold)}")
            ok = False
            continue
        # candidate output columns for each gold column: same values as a column (Spider2's own test)
        gcols, pcols = list(zip(*gold)) if gold else [], list(zip(*pred)) if pred else []
        cands = [[j for j, p in enumerate(pcols) if same_rows([(v,) for v in g], [(v,) for v in p], ordered)[0]]
                 for g in gcols]
        found = False
        for combo in itertools.islice(itertools.product(*cands), 2000):
            if same_rows(gold, [tuple(r[j] for j in combo) for r in pred], ordered)[0]:
                found = True
                break
        print(f"{'ok  ' if found else 'FAIL'} {table}: " +
              ("rows match" if found else "no output columns form rows matching the gold rows"))
        ok &= found
    return ok


def main():
    cfg = json.load(open(sys.argv[1]))
    if cfg["kind"] == "same":
        print("strict = the task's own row-level comparison")
        return 0
    ok = check_gold(cfg) if cfg["kind"] == "gold" else check_spider2(json.load(open(cfg["config"])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
