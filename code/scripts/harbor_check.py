"""Copy the 64 deb analytics tasks into harbor-check/tasks for Harbor's singularity backend.

  python scripts/harbor_check.py

Harbor's singularity environment needs [environment] docker_image (it ignores the Dockerfile), so
62 tasks get the base image and Harbor pulls and converts it itself; the two tasks whose Dockerfile
adds files or packages get a .sif built from it (harbor-check/images). DB_TYPE=duckdb is set for
the agent, solution and verifier phases, as in harness/env.py (the compose default is snowflake).
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "repos/data-eng-bench/tasks"
OUT = ROOT / "harbor-check"
BASE = "ghcr.io/snowflake-labs/data-eng-bench-base:1.0.0"
DEFS = {  # task -> apptainer definition replaying its Dockerfile
    "dbt-consolidate": f"Bootstrap: docker\nFrom: {BASE}\n\n%files\n    {{env}}/data /app/data\n",
    "dbt-session-attribution": f"Bootstrap: docker\nFrom: {BASE}\n\n%post\n    set -e\n"
                               "    pip install --no-cache-dir pandas==2.2.3 numpy==2.2.1\n",
}
ENV = '\n[{}.env]\nDB_TYPE = "duckdb"\n'

ids = sorted(p.name for p in (ROOT / "runs/eval/opus-5-5-t100/deb").iterdir())
assert len(ids) == 64
(OUT / "images").mkdir(parents=True, exist_ok=True)
for t in ids:
    dst = OUT / "tasks" / t
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(TASKS / t, dst, symlinks=True)
    image = BASE
    if t in DEFS:
        image = str(OUT / "images" / f"{t}.sif")
        if not Path(image).exists():
            d = OUT / "images" / f"{t}.def"
            d.write_text(DEFS[t].format(env=TASKS / t / "environment"))
            subprocess.run(["apptainer", "build", "--fakeroot", image, str(d)], check=True)
    toml = (dst / "task.toml").read_text()
    toml = toml.replace("[environment]\n", f'[environment]\ndocker_image = "{image}"\n', 1)
    toml += "".join(ENV.format(s) for s in ["environment", "solution", "verifier"])
    (dst / "task.toml").write_text(toml)
print(f"{len(ids)} tasks in {OUT / 'tasks'}")
