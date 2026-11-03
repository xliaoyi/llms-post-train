"""Add the strict semantic check (strict_check.py) to every task's verifier.

  python harbor-cmp/strict/build_strict.py      # after two oracle runs: validate_trial.sh oracle <task> {oracle,oracle2}

Per task: tests/strict.json (+ tests/strict_gold.duckdb for data-eng-bench, from extract_gold.py over the two oracle
runs), tests/strict_check.py, and a test.sh that runs the original test (renamed test_orig.sh), then the strict check,
and writes reward.json {"reward": the benchmark's own result, "strict": 1 only if both pass,
"strict_check": the strict check alone}.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

H = Path(__file__).resolve().parents[1]
S = H / "strict"
WRAPPER = """#!/bin/bash
# The task's original test (test_orig.sh), then the strict semantic check (strict_check.py; build_strict.py).
# reward.json: reward = the benchmark's own result, strict = 1 only if both pass, strict_check = the strict check
# alone. No reward.txt: no reward.json.
bash /tests/test_orig.sh
rc=$?
# the image's Python (with duckdb) is /usr/local/bin/python3; Harbor's exec PATH finds the system one first
PY=$([ -x /usr/local/bin/python3 ] && echo /usr/local/bin/python3 || echo python3)
$PY /tests/strict_check.py /tests/strict.json > /logs/verifier/strict.txt 2>&1
s=$?
if [ -f /logs/verifier/reward.txt ]; then
  python3 -c "import json, sys; r = float(open('/logs/verifier/reward.txt').read()); json.dump({'reward': r, 'strict': float(r == 1 and sys.argv[1] == '0'), 'strict_check': float(sys.argv[1] == '0')}, open('/logs/verifier/reward.json', 'w'))" "$s"
fi
exit $rc
"""


def deb_gold(task, tests):
    """Write tests/strict_gold.duckdb; return the strict.json config, or None if no stable output tables."""
    db = max(set(found := re.findall(r"/app/[\w/]+\.duckdb", (tests / "test_outputs.py").read_text())),
             key=found.count)
    pristine = json.load(open(H / f"mounts/{task}.json"))["/app"] + db[len("/app"):]
    img = re.search(r'^docker_image\s*=\s*"(.*)"', (H / f"tasks/{task}/task.toml").read_text(), re.M).group(1)
    gold = tests / "strict_gold.duckdb"
    gold.unlink(missing_ok=True)
    p = subprocess.run(["singularity", "exec", "--containall", "-B", f"{S}:/strict", "-B", f"{tests}:/tests",
                        *(["-B", f"{pristine}:/pristine.duckdb:ro"] if Path(pristine).exists() else []), img, "python3",
                        "/strict/extract_gold.py", f"/strict/runs/oracle/{task}.duckdb",
                        f"/strict/runs/oracle2/{task}.duckdb", "/pristine.duckdb" if Path(pristine).exists() else "none",
                        "/tests/test_outputs.py", "/tests/strict_gold.duckdb", "/tests/strict_tables.json"],
                       capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip()[-500:])
    tables = json.load(open(tests / "strict_tables.json")) if p.returncode == 0 else []
    (tests / "strict_tables.json").unlink(missing_ok=True)
    if not tables:
        gold.unlink(missing_ok=True)
        return None
    return {"kind": "gold", "db": db, "tables": tables}


def main():
    summary = {}
    for d in sorted((H / "tasks").iterdir()):
        tests = d / "tests"
        if d.name.startswith("spider2__"):
            cfg = {"kind": "spider2", "config": "/tests/config.json"}
        elif d.name.startswith("ade__"):
            cfg = {"kind": "same"}
        else:
            print(f"== {d.name}")
            cfg = deb_gold(d.name, tests) or {"kind": "same"}
        (tests / "strict.json").write_text(json.dumps(cfg, indent=1))
        shutil.copy(S / "strict_check.py", tests / "strict_check.py")
        if not (tests / "test_orig.sh").exists():
            (tests / "test.sh").rename(tests / "test_orig.sh")
        (tests / "test.sh").write_text(WRAPPER)
        summary[d.name] = cfg["kind"] + (f" ({len(cfg['tables'])} tables)" if cfg["kind"] == "gold" else "")
    json.dump(summary, open(S / "strict_summary.json", "w"), indent=1)
    kinds = [v.split()[0] for v in summary.values()]
    print({k: kinds.count(k) for k in set(kinds)})


if __name__ == "__main__":
    main()
