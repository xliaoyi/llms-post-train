"""Offline mirror of the dbt packages the tasks use, so `dbt deps` works while agents have no internet.

  python harbor-cmp/dbt_mirror.py build     # needs internet: writes harbor-cmp/dbt-mirror/
  python harbor-cmp/dbt_mirror.py serve     # http://127.0.0.1:8110/ (run_trial.sh sets DBT_PACKAGE_HUB_URL to it)

build: for every hub package in the tasks' packages.yml and, recursively, the packages its versions depend on,
saves the hub's api/v1/<org>/<name>.json with each version's tarball URL pointed at the mirror, the tarballs,
and the hub's index.json restricted to the mirrored packages.
Git packages are cloned bare into dbt-mirror/git/ (run_trial.sh mounts it at /opt/dbt-mirror/git and redirects
their URLs there with git's GIT_CONFIG_* environment).
"""
import json
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

H = Path(__file__).resolve().parent
OUT = H / "dbt-mirror"
HUB = "https://hub.getdbt.com/"
PORT = 8110
LOCAL = f"http://127.0.0.1:{PORT}/"


def task_packages():
    hub, git = set(), set()
    for m in (H / "mounts").glob("*.json"):
        for root in json.load(open(m)).values():
            for f in Path(root).rglob("packages.yml"):
                if "dbt_packages" in f.parts:
                    continue
                text = f.read_text()
                hub |= set(re.findall(r"package:\s*['\"]?([\w.-]+/[\w.-]+)", text))
                git |= set(re.findall(r"git:\s*['\"]?([^'\"\s]+)", text))
    return hub, git


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def build():
    hub, git = task_packages()
    seen, todo, tarballs = set(), sorted(hub), []
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            meta = json.loads(fetch(f"{HUB}api/v1/{name}.json"))
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        if meta.get("redirectnamespace"):  # renamed package: mirror the new name too
            todo.append(f"{meta['redirectnamespace']}/{meta.get('redirectname') or name.split('/')[1]}")
        for ver, info in meta.get("versions", {}).items():
            url = info["downloads"]["tarball"]
            local = f"tarballs/{name.replace('/', '__')}__{ver}.tar.gz"
            tarballs.append((url, OUT / local))
            info["downloads"]["tarball"] = LOCAL + local
            todo += [p["package"] for p in info.get("packages", []) if "package" in p and p["package"] not in seen]
        dst = OUT / f"api/v1/{name}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(meta))
    # dbt checks a package name against the hub index first; list only the mirrored packages
    (OUT / "api/v1/index.json").write_text(json.dumps([n for n in json.loads(fetch(f"{HUB}api/v1/index.json")) if n in seen]))
    (OUT / "tarballs").mkdir(parents=True, exist_ok=True)

    def get(job):
        url, dst = job
        if not dst.exists():
            try:
                dst.write_bytes(fetch(url))
            except Exception as e:
                print(f"tarball failed {url}: {e}")

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(get, tarballs))
    (OUT / "git").mkdir(exist_ok=True)
    for url in sorted(git):
        dst = OUT / "git" / (url.rstrip("/").split("/")[-1].removesuffix(".git") + ".git")
        if not dst.exists():
            subprocess.run(["git", "clone", "--bare", "-q", url, str(dst)], check=True)
    print(f"{len(seen)} hub packages, {len(tarballs)} tarballs, {len(git)} git repos in {OUT}")


def serve():
    handler = partial(SimpleHTTPRequestHandler, directory=str(OUT))
    ThreadingHTTPServer(("127.0.0.1", PORT), handler).serve_forever()


if __name__ == "__main__":
    {"build": build, "serve": serve}[sys.argv[1]]()
