#!/bin/bash
# One Harbor trial: config $1 (qwen-think | qwen-nothink | alt-think | alt-nothink), task $2 (<source>__<id>),
# attempt $3, split $4. Mounts a fresh copy of the task's pristine workspace (Harbor's singularity backend
# ignores Dockerfiles); Harbor runs the agent and the task's verifier. Results: harbor-cmp/jobs/<config>/<split>/.
#   bash harbor-cmp/run_trial.sh alt-think deb__dbt-abc-classification 0 test
CFG=$1 TASK=$2 ATTEMPT=$3 SPLIT=$4
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
export PYTHONPATH=$PWD/harbor-cmp
H=$PWD/harbor-cmp
case $CFG in  # qwen-*: GPU0 vLLM :8008 via proxies 8100/8101; alt-*: GPU1 vLLM :8009 via proxies 8102/8103
  qwen-think)   PORT=8100 ;;
  qwen-nothink) PORT=8101 ;;
  alt-think)    PORT=8102 ;;
  alt-nothink)  PORT=8103 ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
# The proxy logs every model request/response of this trial to proxy-logs/<run id>.jsonl
URL=http://127.0.0.1:$PORT/r/$CFG--$SPLIT--$TASK--$ATTEMPT/v1
# No internet for the agent: HTTP clients (pip, npm, curl, wget, Node, Bun) go through a dead proxy; only the
# model endpoint on 127.0.0.1 is reachable. The agents are installed offline from read-only mounts.
DEAD=http://127.0.0.1:9
OFFLINE=(--ae HTTP_PROXY=$DEAD --ae HTTPS_PROXY=$DEAD --ae http_proxy=$DEAD --ae https_proxy=$DEAD
         --ae ALL_PROXY=$DEAD --ae all_proxy=$DEAD --ae NO_PROXY=127.0.0.1,localhost --ae no_proxy=127.0.0.1,localhost
         --ae PIP_NO_INDEX=1 --ae npm_config_offline=true
         # `dbt deps` uses the offline package mirror (dbt_mirror.py serve); the one git package, a local bare clone
         --ae DBT_PACKAGE_HUB_URL=http://127.0.0.1:8110/ --ae GIT_CONFIG_COUNT=1
         --ae GIT_CONFIG_KEY_0=url./opt/dbt-mirror/git/dbt-duckdb-utils.git.insteadOf
         --ae GIT_CONFIG_VALUE_0=https://github.com/sdebruyn/dbt-duckdb-utils.git)
case $CFG in
  qwen-*) AGENT=(-a qwen_agent:QwenCodePinned -m openai/student --ae OPENAI_BASE_URL=$URL --ae OPENAI_API_KEY=unused
                 --ae QWEN_STREAM_IDLE_TIMEOUT_MS=0 --ak config=$H/qwen-settings.json) ;;
  alt-*)  AGENT=(-a altimate_agent:AltimateCode -m vllm/student --ae ALTIMATE_BASE_URL=$URL --ae ALTIMATE_MAX_TURNS=100
                 --ae ALTIMATE_BINARY=/opt/altimate-dist/altimate) ;;
esac
# Anonymous workspace path: the bind source is visible inside the container (/proc/mounts)
mkdir -p /tmp/harbor-ws-$USER
WS=$(mktemp -d /tmp/harbor-ws-$USER/XXXXXXXXXX)
MOUNTS=$(python3 - "$H/mounts/$TASK.json" "$WS" "$H/agents-dist" "$PWD/agents" "$H/dbt-mirror/git" <<'EOF'
import json, shutil, sys
src, ws, dist, agents, git = sys.argv[1:]
mounts = []
for target, pristine in json.load(open(src)).items():
    dst = f"{ws}{target}"
    shutil.copytree(pristine, dst, symlinks=True)
    mounts.append({"type": "bind", "source": dst, "target": target})
mounts.append({"type": "bind", "source": dist, "target": "/opt/altimate-dist", "read_only": True})
mounts.append({"type": "bind", "source": agents, "target": "/opt/agents", "read_only": True})
mounts.append({"type": "bind", "source": git, "target": "/opt/dbt-mirror/git", "read_only": True})
print(json.dumps(mounts))
EOF
)
OUT=$H/jobs/$CFG/$SPLIT
mkdir -p $OUT
rm -rf $OUT/$TASK-$ATTEMPT
.venv-harbor25/bin/harbor run -p $H/tasks/$TASK -e tacc_env:TaccSingularityEnvironment "${AGENT[@]}" "${OFFLINE[@]}" \
  -y -q -o $OUT --job-name $TASK-$ATTEMPT --mounts "$MOUNTS" > $OUT/$TASK-$ATTEMPT.log 2>&1
rc=$?
# Harbor's singularity backend leaves the container running after the trial (its starter is re-parented to init),
# with the agent's processes still holding memory. Containers of running trials keep their `singularity exec`
# parent, so any orphaned starter belongs to a finished trial.
for p in $(ps -u $USER -o pid=,ppid=,comm= | awk '$2==1 && $3=="starter" {print $1}'); do kill $p 2>/dev/null; done
rm -rf $WS
exit $rc
