"""Validate every task before sampling; writes harbor-cmp/validated_tasks.json, which driver.py requires.

  python harbor-cmp/preflight.py [-j 16]

Per task, kept separate so setup failures are not confused with task defects:
  setup:   in the task image with the pristine workspace mounted read-only, the required tools (dbt, python3,
           duckdb) run and every DuckDB database in the workspace opens.
  oracle:  the reference solution passes from a clean start, on the benchmark's own grader and the strict check.
  nop:     the untouched task fails.
  excluded_tasks.json: known task defects, with the reason.
Also records the snapshot each trial starts from: sha256 of the image and of the pristine workspace.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
SING = "/opt/apps/tacc-apptainer/1.4.1/bin/singularity"
CHECK = """set -e
command -v dbt python3 >/dev/null
dbt --version | head -3
python3 - <<'EOF'
import duckdb, glob
for d in {dirs}:
    for f in glob.glob(d + '/**/*.duckdb', recursive=True):
        duckdb.connect(f, read_only=True).execute('select 1').fetchall()
        print('opened', f)
EOF
"""


def sha256_file(path, cache={}):
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        cache[path] = h.hexdigest()
    return cache[path]


def sha256_tree(root):
    """Hash of every file's relative path and contents (symlinks by target)."""
    h = hashlib.sha256()
    for dirpath, dirs, files in sorted(os.walk(root)):
        dirs.sort()
        for name in sorted(files):
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).encode() + b"\0")
            h.update(os.readlink(p).encode() if os.path.islink(p) else bytes.fromhex(sha256_file(p, {})))
    return h.hexdigest()


def setup_check(task, image, mounts):
    binds = sum([["-B", f"{src}:{dst}:ro"] for dst, src in mounts.items()], [])
    p = subprocess.run([SING, "exec", "--containall", *binds, image, "bash", "-c",
                        CHECK.format(dirs=list(mounts))], capture_output=True, text=True, timeout=600)
    return p.returncode == 0, (p.stdout + p.stderr).strip()[-1500:]


def trial(run, task):
    """(reward, strict) of a validation trial, or None."""
    found = list((H / f"jobs/validate-{run}/{task}").glob("*/result.json"))
    if not found:
        return None
    rewards = (json.loads(found[0].read_text()).get("verifier_result") or {}).get("rewards") or {}
    return rewards.get("reward"), rewards.get("strict")


def validate(task):
    toml = (H / f"tasks/{task}/task.toml").read_text()
    image = re.search(r'^docker_image\s*=\s*"(.*)"', toml, re.M).group(1)
    mounts = json.load(open(H / f"mounts/{task}.json"))
    for agent in ["oracle", "nop"]:
        subprocess.run(["bash", str(H / "validate_trial.sh"), agent, task, f"final-{agent}"])
    setup_ok, setup_log = setup_check(task, image, mounts)
    oracle, nop = trial("final-oracle", task), trial("final-nop", task)
    excluded = json.load(open(H / "excluded_tasks.json")).get(task)
    rec = {"setup_ok": setup_ok, "setup_log": setup_log, "oracle": oracle, "nop": nop, "excluded": excluded,
           "image": image, "image_sha256": sha256_file(image),
           "workspace_sha256": {dst: sha256_tree(src) for dst, src in mounts.items()}}
    rec["ok"] = bool(setup_ok and oracle == (1.0, 1.0) and nop and nop[0] == 0 and not excluded)
    print(f"{task}: ok={rec['ok']} setup={setup_ok} oracle={oracle} nop={nop}" + (" EXCLUDED" if excluded else ""),
          flush=True)
    return task, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=16)
    args = ap.parse_args()
    tasks = sorted(p.name for p in (H / "tasks").iterdir())
    with ThreadPoolExecutor(args.j) as pool:
        out = dict(pool.map(validate, tasks))
    json.dump(out, open(H / "validated_tasks.json", "w"), indent=1)
    print(f"{sum(r['ok'] for r in out.values())}/{len(out)} tasks validated")


if __name__ == "__main__":
    main()
