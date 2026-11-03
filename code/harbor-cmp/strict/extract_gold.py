"""Save the reference solution's output tables for a data-eng-bench task (run inside the task image: needs duckdb).

  python3 extract_gold.py <run1.duckdb> <run2.duckdb> <pristine.duckdb> <test_outputs.py> <out.duckdb> <out.json>

run1/run2 are the task database after two independent oracle trials (solution + verifier). Output tables are those
the grader names that are new or changed relative to the pristine database. Columns whose values differ between
the two runs (timestamps, random ids) are left out; a table whose rows still differ is left out entirely.
out.json lists [{schema, name, columns}] for strict_check.py.
"""
import json
import re
import sys

import duckdb

sys.path.insert(0, "/strict")
from strict_check import same_rows  # noqa: E402

run1, run2, pristine, tests, out_db, out_json = sys.argv[1:]
con = duckdb.connect(":memory:")
for alias, path in [("r1", run1), ("r2", run2), ("p", pristine)]:
    if path == "none":  # the database does not exist before the solution runs: every table is new
        con.execute(f"ATTACH ':memory:' AS {alias}")
    else:
        con.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")


def tables(alias):
    return {(s.lower(), n.lower()): (s, n) for s, n in con.execute(
        "SELECT table_schema, table_name FROM information_schema.tables WHERE table_catalog = ?", [alias]).fetchall()}


def unchanged(a, b):
    """Same rows (as a multiset) in two tables; False if they cannot be compared (different columns)."""
    try:
        return all(con.execute(f"SELECT count(*) FROM (SELECT * FROM {x} EXCEPT ALL SELECT * FROM {y})").fetchone()[0] == 0
                   for x, y in [(a, b), (b, a)])
    except duckdb.Error:
        return False


src = open(tests).read()
refs = set(re.findall(r"[a-z_][a-z0-9_]*", src.lower()))  # any name in the grader (SQL or constants)
r1, r2, pr = tables("r1"), tables("r2"), tables("p")
con.execute(f"ATTACH '{out_db}' AS g")
kept = []
for k, (s, n) in sorted(r1.items()):
    if n.lower() not in refs or k not in r2:
        continue
    if k in pr and unchanged('p."{}"."{}"'.format(*pr[k]), f'r1."{s}"."{n}"'):
        continue  # input table, unchanged by the solution
    cols = [c for c, in con.execute("SELECT column_name FROM information_schema.columns WHERE table_catalog = 'r1' "
                                    "AND table_schema = ? AND table_name = ? ORDER BY ordinal_position", [s, n]).fetchall()]
    a = con.execute(f'SELECT * FROM r1."{s}"."{n}"').fetchall()
    b = con.execute(f'SELECT * FROM r2."{s}"."{n}"').fetchall()
    stable = [c for i, c in enumerate(cols) if same_rows([(r[i],) for r in a], [(r[i],) for r in b])[0]]
    sel = ", ".join(f'"{c}"' for c in stable)
    if not stable or not same_rows(con.execute(f'SELECT {sel} FROM r1."{s}"."{n}"').fetchall(),
                                   con.execute(f'SELECT {sel} FROM r2."{s}"."{n}"').fetchall())[0]:
        print(f"skip {s}.{n}: reference runs disagree")
        continue
    con.execute(f'CREATE SCHEMA IF NOT EXISTS g."{s}"')
    con.execute(f'CREATE TABLE g."{s}"."{n}" AS SELECT {sel} FROM r1."{s}"."{n}"')
    kept.append({"schema": s, "name": n, "columns": stable})
    dropped = [c for c in cols if c not in stable]
    print(f"gold {s}.{n}: {len(a)} rows, {len(stable)} columns" + (f" (unstable, skipped: {dropped})" if dropped else ""))
json.dump(kept, open(out_json, "w"), indent=1)
