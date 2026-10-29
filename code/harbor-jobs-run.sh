#!/bin/bash
# Convert and publicly upload the per-model train/eval jobs given as arguments.
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib
for j in "$@"; do
  .venv-harbor/bin/python -u scripts/to_harbor_jobs.py $j || { echo "$j: convert failed"; continue; }
  COLUMNS=200 .venv-harbor/bin/harbor upload harbor-jobs/$j --public --share inf385t-llms-post-train > harbor-jobs/$(echo $j | tr / -)-upload.log 2>&1
  echo "$j: $(grep -E 'Uploaded|View at|Error' harbor-jobs/$(echo $j | tr / -)-upload.log | tr '\n' ' ')"
done
echo ALLDONE
