#!/bin/bash
# Run one Harbor trial (singularity backend) of agent $1 (oracle|nop) on task $2 from harbor-check/tasks.
#   ls harbor-check/tasks | xargs -P 8 -I{} bash scripts/harbor_check.sh oracle {}
# /app is a fresh host copy of the image's /app bound over it: deleting image directories on
# --writable-tmpfs fails with EIO on TACC, which breaks `dbt deps` in the reference solutions.
AGENT=$1 TASK=$2
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
H=$PWD/harbor-check
IMAGE=$(sed -n 's/^docker_image = "\(.*\)"/\1/p' $H/tasks/$TASK/task.toml)
case $IMAGE in
  *.sif) SIF=$IMAGE ;;
  *) SIF=$H/sif-cache/$(echo "${IMAGE}" | tr '/:' '__').sif ;;  # Harbor's cache name for the image
esac
WORK=$H/work/$AGENT/$TASK
rm -rf $WORK && mkdir -p $WORK
apptainer exec --fakeroot $SIF cp -a /app $WORK/app
rm -rf $H/jobs/$AGENT/$TASK
.venv-harbor/bin/harbor run -p $H/tasks/$TASK -a $AGENT -e singularity -y -q \
  --ek singularity_image_cache_dir=$H/sif-cache -o $H/jobs/$AGENT --job-name $TASK \
  --mounts-json "[{\"type\": \"bind\", \"source\": \"$WORK/app\", \"target\": \"/app\"}]" \
  > $H/jobs/$AGENT-$TASK.log 2>&1
rm -rf $WORK
