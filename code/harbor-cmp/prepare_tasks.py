"""Build Harbor task dirs for the v3 split and the pristine workspaces their trials mount.

  python harbor-cmp/prepare_tasks.py

For each task in data/{train,test}_tasks_v3.json: copy the task to harbor-cmp/tasks/<source>__<id> and set
[environment] docker_image to our local Apptainer image (Harbor's singularity backend ignores Dockerfiles), a
uniform 2-hour agent timeout and 32 GB memory limit, and DB_TYPE=duckdb. The project state the Dockerfile would build (copied project,
dbt deps, ADE setup.sh) comes from harness/env.py's pristine cache; run_trial.sh mounts a fresh copy of it per
trial (harbor-cmp/mounts/<task>.json). ADE tests need ade_bench_equality_test.sql in /app/macros at verification (env.py copies it before
test.sh), so it is shipped in tests/ade_macros and test.sh copies it in. Then run preflight.py (validation) and
strict/build_strict.py (strict semantic check).
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from env import MOUNTS, Task  # noqa: E402

OUT = ROOT / "harbor-cmp/tasks"
MOUNT_DIR = ROOT / "harbor-cmp/mounts"  # <task>.json: container path -> pristine host dir
AGENT_TIMEOUT = 7200.0
# Harbor's singularity backend aborts the whole trial (no verifier) when the container exceeds memory_mb (4-8 GB in
# the tasks); the agent CLI itself counts against it. Use one generous limit for every task and harness.
MEMORY_MB = 32768
# Defects found by validation (reference solution failing from a clean start): (file, old, new) per task
TASK_FIXES = {
    # the instruction and data put the source models in main; the grader read them from the output schema
    "deb__dbt-fix-marketing-attribution": [(
        "tests/test_outputs.py", "SOURCE_SCHEMA = SCHEMA\n",
        "SOURCE_SCHEMA = 'main' if SCHEMA.lower() == 'analytics' else SCHEMA  # sources materialize in main (see instruction)\n")],
    # the DuckDB profile is written to ~/.dbt but DBT_PROFILES_DIR still pointed at the reference project
    "deb__dbt-product-return-rate-analysis": [(
        "solution/solve.sh", "      schema: analytics\nEOF\nfi\n",
        "      schema: analytics\nEOF\n    export DBT_PROFILES_DIR=\"$HOME/.dbt\"\nfi\n")],
}


def harbor_image(sif):
    """Harbor's singularity server needs /usr/bin/python3; the python:3.11-slim based images (Spider2, ADE) only have
    /usr/local/bin/python3, so use a copy with the symlink (harbor-cmp/images/<name>-harbor.def)."""
    patched = ROOT / "harbor-cmp/images" / f"{sif.stem}-harbor.sif"
    return patched if sif.stem in ("spider2", "adebench") else sif


def set_key(toml, section, key, value):
    """Set key = value in [section], adding the section if absent."""
    m = re.search(rf"^\[{re.escape(section)}\]\n", toml, flags=re.M)
    if not m:
        return toml.rstrip("\n") + f"\n\n[{section}]\n{key} = {value}\n"
    body_end = re.search(r"^\[", toml[m.end():], flags=re.M)
    end = m.end() + (body_end.start() if body_end else len(toml) - m.end())
    body = re.sub(rf"^{key}\s*=.*\n", "", toml[m.end():end], flags=re.M)
    return toml[:m.end()] + f"{key} = {value}\n" + body + toml[end:]


def main():
    tasks = [t for f in ["train", "test"] for t in json.load(open(ROOT / f"data/{f}_tasks_v3.json"))]
    OUT.mkdir(parents=True, exist_ok=True)
    MOUNT_DIR.mkdir(exist_ok=True)
    for t in tasks:
        task = Task(t["source"], t["task"])
        task.ensure_pristine()
        dst = OUT / f"{t['source']}__{t['task']}"
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir()
        for name in ["instruction.md", "task.toml", "tests", "solution"]:
            src = task.dir / name
            if src.is_dir():
                shutil.copytree(src, dst / name, symlinks=True)
            elif src.exists():
                shutil.copy2(src, dst / name)
        # Harbor's singularity backend reads only WORKDIR from the Dockerfile (default /app); Spider2 uses /workspace
        (dst / "environment").mkdir()
        (dst / "environment/Dockerfile").write_text(f"FROM {task.sif}\nWORKDIR {task.workdir}\n")
        toml = (dst / "task.toml").read_text()
        toml = set_key(toml, "environment", "docker_image", json.dumps(str(harbor_image(task.sif))))
        toml = set_key(toml, "agent", "timeout_sec", AGENT_TIMEOUT)
        toml = set_key(toml, "environment", "memory_mb", MEMORY_MB)
        for section in ["environment.env", "solution.env", "verifier.env"]:
            toml = set_key(toml, section, "DB_TYPE", '"duckdb"')
        (dst / "task.toml").write_text(toml)
        if t["source"] == "ade":
            macros = ROOT / "repos/ade-bench/tasks" / task.source_id() / "macros"
            if macros.exists():
                shutil.copytree(macros, dst / "tests/ade_macros")
                test = (dst / "tests/test.sh").read_text()
                test = test.replace("cd /app\n", "cd /app\nmkdir -p macros && cp /tests/ade_macros/* macros/\n", 1)
                (dst / "tests/test.sh").write_text(test)
            # ADE's solution.sh reads patches from its own mount point; Harbor uploads them under /solution
            sol = dst / "solution/solution.sh"
            sol.write_text(sol.read_text().replace("/sage/solutions/", "/solution/solutions/"))
        for path, old, new in TASK_FIXES.get(dst.name, []):
            text = (dst / path).read_text()
            assert text.count(old) == 1, (dst.name, path)
            (dst / path).write_text(text.replace(old, new))
        mounts = {m: str(task.pristine / d) for d, m in MOUNTS[t["source"]].items()}
        (MOUNT_DIR / f"{dst.name}.json").write_text(json.dumps(mounts))
    print(f"{len(tasks)} tasks in {OUT}")


if __name__ == "__main__":
    main()
