#!/bin/bash
# Validation trial without a model: agent $1 (oracle | nop), task $2, run name $3 (default: $1). Same environment,
# mounts and verifier as run_trial.sh ($TASKS: task dir root, default harbor-cmp/tasks; $MOUNT: mount map name,
# default the task). Results: harbor-cmp/jobs/validate-<run>/<task>/. For data-eng-bench oracle
# runs, the tables the grader queries are saved to harbor-cmp/strict/runs/<run>/<task>.duckdb (extract_gold.py).
#   bash harbor-cmp/validate_trial.sh oracle deb__dbt-abc-classification oracle2
AG=$1 TASK=$2 RUN=${3:-$1}
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
export PYTHONPATH=$PWD/harbor-cmp
H=$PWD/harbor-cmp
TASKS=${TASKS:-$H/tasks} MOUNT=${MOUNT:-$TASK}
mkdir -p /tmp/harbor-ws-$USER
WS=$(mktemp -d /tmp/harbor-ws-$USER/XXXXXXXXXX)
MOUNTS=$(python3 - "$H/mounts/$MOUNT.json" "$WS" <<'EOF'
import json, shutil, sys
src, ws = sys.argv[1:]
mounts = []
for target, pristine in json.load(open(src)).items():
    shutil.copytree(pristine, ws + target, symlinks=True)
    mounts.append({"type": "bind", "source": ws + target, "target": target})
print(json.dumps(mounts))
EOF
)
OUT=$H/jobs/validate-$RUN
mkdir -p $OUT
rm -rf $OUT/$TASK
.venv-harbor25/bin/harbor run -p $TASKS/$TASK -e tacc_env:TaccSingularityEnvironment -a $AG -y -q \
  -o $OUT --job-name $TASK --mounts "$MOUNTS" > $OUT/$TASK.log 2>&1
if [ $AG = oracle ] && [[ $TASK == deb__* ]]; then
  DB=$(grep -oh "/app/[A-Za-z0-9_/]*\.duckdb" $TASKS/$TASK/tests/test_outputs.py | sort | uniq -c | sort -rn | awk 'NR==1{print $2}')
  IMG=$(sed -n 's/^docker_image *= *"\(.*\)"/\1/p' $TASKS/$TASK/task.toml)
  mkdir -p $H/strict/runs/$RUN && rm -f $H/strict/runs/$RUN/$TASK.duckdb
  singularity exec --containall -B $WS/app:/app -B $TASKS/$TASK/tests:/tests:ro -B $H/strict:/strict $IMG \
    python3 /strict/snapshot.py $DB /tests/test_outputs.py /strict/runs/$RUN/$TASK.duckdb >> $OUT/$TASK.log 2>&1
fi
for p in $(ps -u $USER -o pid=,ppid=,comm= | awk '$2==1 && $3=="starter" {print $1}'); do kill $p 2>/dev/null; done
rm -rf $WS
