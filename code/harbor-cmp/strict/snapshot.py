"""Copy the tables a grader names out of a trial's database (run inside the task image: needs duckdb).

  python3 snapshot.py <task.duckdb> <test_outputs.py> <out.duckdb>

Keeps reference-solution outputs for extract_gold.py without storing the whole (~470 MB) database per run.
"""
import re
import sys

import duckdb

db, tests, out = sys.argv[1:]
refs = set(re.findall(r"[a-z_][a-z0-9_]*", open(tests).read().lower()))  # any name in the grader (SQL or constants)
con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{db}' AS src (READ_ONLY)")
con.execute(f"ATTACH '{out}' AS dst")
for s, n in con.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_catalog = 'src'").fetchall():
    if n.lower() in refs:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS dst."{s}"')
        con.execute(f'CREATE TABLE dst."{s}"."{n}" AS SELECT * FROM src."{s}"."{n}"')
