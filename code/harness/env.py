"""Docker-free task environments for data-eng-bench, Spider2-DBT and ADE-bench.

Each task runs in an Apptainer image with a per-run writable copy of its project
directory bind-mounted at the path the task expects (/app or /workspace).
Tests and reference solutions are only mounted for verify() / oracle(), never
while the agent works.

Layout (under ROOT):
  images/*.sif               container images
  cache/pristine/<src>/<id>  prepared task state (persistent)
  runs/...                   logs and results (persistent)
Per-run working copies live on node-local disk (WORK) and are deleted on close().
"""
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import tomllib
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("DBTENV_WORK", f"/tmp/dbtenv-{os.environ['USER']}"))
APPTAINER = os.environ.get("APPTAINER", "apptainer")

SOURCES = {
    "deb": ROOT / "repos/data-eng-bench/tasks",
    "spider2": ROOT / "datasets/spider2-dbt",
    "ade": ROOT / "datasets/ade-bench",
}
# Bind layout per source: local dir name -> container mount point.
MOUNTS = {
    "deb": {"app": "/app"},
    "spider2": {"workspace": "/workspace"},
    "ade": {"app": "/app", "scripts": "/scripts"},
}
WORKDIR = {"deb": "/app", "spider2": "/workspace", "ade": "/app"}
DEB_PIP_IMAGES = {  # tasks whose Dockerfile pip-installs extra packages
    "pandas==2.2.3 numpy==2.2.1": "deb-pd223-np221.sif",
    "pandas==2.2.3": "deb-pd223.sif",
}


def list_tasks(source, category=None):
    ids = sorted(p.name for p in SOURCES[source].iterdir() if (p / "task.toml").exists())
    if category:
        ids = [t for t in ids if task_meta(source, t).get("metadata", {}).get("category") == category]
    return ids


def task_meta(source, task_id):
    return tomllib.loads((SOURCES[source] / task_id / "task.toml").read_text())


def _run(cmd, log=None, timeout=None, check=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       errors="replace", timeout=timeout)
    if log:
        with open(log, "a") as f:
            f.write(f"$ {shlex.join(cmd)}\n{p.stdout}\n[exit {p.returncode}]\n")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {shlex.join(cmd)}\n{p.stdout[-2000:]}")
    return p


class Task:
    def __init__(self, source, task_id):
        self.source, self.id = source, task_id
        self.dir = SOURCES[source] / task_id
        self.meta = task_meta(source, task_id)
        self.instruction = (self.dir / "instruction.md").read_text()
        self.workdir = WORKDIR[source]
        self.sif = ROOT / "images" / self._image()
        self.pristine = ROOT / "cache/pristine" / source / task_id

    def _image(self):
        if self.source == "spider2":
            return "spider2.sif"
        if self.source == "ade":
            return "adebench.sif"
        dockerfile = (self.dir / "environment/Dockerfile").read_text()
        for pkgs, sif in DEB_PIP_IMAGES.items():
            if all(p in dockerfile for p in pkgs.split()):
                return sif
        return "deb-base.sif"

    def env_vars(self, section):
        env = {"DB_TYPE": "duckdb"}
        env.update(self.meta.get(section, {}).get("env", {}))
        return env

    # ---------- pristine state (built once per task, cached) ----------
    def ensure_pristine(self):
        self.pristine.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{self.pristine}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (self.pristine / ".ready").exists():
                return
            if self.pristine.exists():
                shutil.rmtree(self.pristine)
            tmp = self.pristine.with_name(self.pristine.name + ".building")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            log = tmp / "prepare.log"
            getattr(self, f"_prepare_{self.source}")(tmp, log)
            (tmp / ".ready").touch()
            tmp.rename(self.pristine)

    def _exec_build(self, root, cmd, log, extra_binds=()):
        binds = [f"{root / d}:{m}" for d, m in MOUNTS[self.source].items()] + list(extra_binds)
        (root / "home").mkdir(exist_ok=True)  # persistent $HOME (=/root, as in Docker)
        args = [APPTAINER, "-q", "exec", "--containall", "--writable-tmpfs", "--pwd", self.workdir,
                "--home", f"{root / 'home'}:/root"]
        for b in binds:
            args += ["-B", b]
        for k, v in self.env_vars("environment").items():
            args += ["--env", f"{k}={v}"]
        return _run(args + [str(self.sif), "bash", "-c", cmd], log=log, check=False)

    def _prepare_deb(self, root, log):
        base = ROOT / "cache/deb-app"
        if not base.exists():
            _run([APPTAINER, "exec", str(ROOT / "images/deb-base.sif"), "cp", "-a", "/app", str(base)])
        shutil.copytree(base, root / "app", symlinks=True)
        env_dir = self.dir / "environment"
        # Replay the task Dockerfile on top of the base /app (only COPY / RUN / WORKDIR occur).
        text = re.sub(r"\\\n", " ", (env_dir / "Dockerfile").read_text())
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("COPY "):
                *srcs, dst = line.split()[1:]
                assert dst.startswith("/app"), line
                target = root / "app" / dst[len("/app/"):]
                for s in srcs:
                    src = env_dir / s
                    if src.is_dir():
                        shutil.copytree(src, target, dirs_exist_ok=True)
                    else:
                        dest = target / src.name if (dst.endswith("/") or len(srcs) > 1) else target
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
            elif line.startswith("RUN ") and "pip install" not in line:
                p = self._exec_build(root, line[4:], log)
                if p.returncode != 0:
                    raise RuntimeError(f"{self.id}: Dockerfile RUN failed, see {log}")

    def _prepare_spider2(self, root, log):
        shutil.copytree(self.dir / "environment/dbt_project", root / "workspace", symlinks=True)
        if (root / "workspace/packages.yml").exists():
            self._exec_build(root, "dbt deps --profiles-dir /workspace", log)

    def _prepare_ade(self, root, log):
        env_dir = self.dir / "environment"
        app = root / "app"
        shutil.copytree(env_dir / "project", app, symlinks=True)
        for f in ["CLAUDE.md", "GEMINI.md", "AGENTS.md", "MACRO.md", "setup.sh"]:
            shutil.copy2(env_dir / f, app / f)
        shutil.copytree(env_dir / "setup-data", app / "setup", dirs_exist_ok=True)
        shutil.copytree(env_dir / "shared-scripts", root / "scripts", dirs_exist_ok=True)
        db_name = (env_dir / "db_name.txt").read_text().strip()
        db = ROOT / "cache/ade-db" / f"{db_name}.duckdb"
        if not db.exists():
            db.parent.mkdir(parents=True, exist_ok=True)
            file_id = (env_dir / "db_file_id.txt").read_text().strip()
            _run([APPTAINER, "exec", str(self.sif), "gdown", "-q",
                  f"https://drive.google.com/uc?id={file_id}", "-O", f"{db}.part"], log=log)
            os.rename(f"{db}.part", db)
        shutil.copy2(db, app / f"{db_name}.duckdb")
        # setup.sh plants the task's bug; it may exit non-zero because the planted bug breaks dbt run.
        self._exec_build(root, "bash setup.sh --db-type=duckdb --project-type=dbt", log)
        (app / "setup.sh").unlink()
        shutil.rmtree(app / "setup")

    def source_id(self):
        return self.meta["task"]["name"].split("__")[1]


class Env:
    """One run of one task. exec() is what the agent's tools call."""

    def __init__(self, task: Task, out_dir):
        self.task = task
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        task.ensure_pristine()
        # Mirror the pristine state to node-local disk once; per-run copies come from there.
        local = WORK / "pristine" / task.source / task.id
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{local}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not local.exists():
                tmp = local.with_name(local.name + ".copying")
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.copytree(task.pristine, tmp, symlinks=True)
                tmp.rename(local)
        self.root = WORK / "runs" / uuid.uuid4().hex[:12]
        shutil.copytree(local, self.root, symlinks=True)
        (self.root / "logs/verifier").mkdir(parents=True, exist_ok=True)
        (self.root / "home").mkdir(exist_ok=True)

    def _args(self, extra_binds=(), env_section="environment"):
        args = [APPTAINER, "-q", "exec", "--containall", "--writable-tmpfs", "--pwd", self.task.workdir,
                "--home", f"{self.root / 'home'}:/root"]
        for d, m in MOUNTS[self.task.source].items():
            args += ["-B", f"{self.root / d}:{m}"]
        args += ["-B", f"{self.root / 'logs'}:/logs"]
        for b in extra_binds:
            args += ["-B", b]
        for k, v in self.task.env_vars(env_section).items():
            args += ["--env", f"{k}={v}"]
        return args + [str(self.task.sif)]

    def exec(self, cmd, timeout=180):
        """Run a shell command as the agent. Returns (exit_code, output)."""
        try:
            p = subprocess.run(self._args() + ["bash", "-c", cmd], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout)
            return p.returncode, p.stdout
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return 124, out + f"\n[timed out after {timeout}s]"

    def oracle(self):
        sol = self.task.dir / "solution"
        binds = [f"{sol}:/solution:ro"]
        if (sol / "solutions").exists():
            binds.append(f"{sol / 'solutions'}:/sage/solutions:ro")
        _run(self._args(binds, "solution") + ["bash", "/solution/solve.sh"],
             log=self.out / "oracle.log", check=False, timeout=3600)

    def verify(self):
        """Run the official verifier on the current state. Returns reward (0 or 1)."""
        if self.task.source == "ade":
            macros = ROOT / "repos/ade-bench/tasks" / self.task.source_id() / "macros"
            if macros.exists():
                shutil.copytree(macros, self.root / "app/macros", dirs_exist_ok=True)
        timeout = self.task.meta.get("verifier", {}).get("timeout_sec", 1800)
        try:
            _run(self._args([f"{self.task.dir / 'tests'}:/tests:ro"], "verifier") + ["bash", "/tests/test.sh"],
                 log=self.out / "verify.log", check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            (self.out / "verify.log").open("a").write(f"\n[verifier timed out after {timeout}s]\n")
        reward_file = self.root / "logs/verifier/reward.txt"
        reward = int(reward_file.read_text().strip() or 0) if reward_file.exists() else 0
        (self.out / "reward.txt").write_text(f"{reward}\n")
        return reward

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)
