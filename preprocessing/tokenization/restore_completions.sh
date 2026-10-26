#!/bin/bash
# Restore datatrove completion markers (+ executor.json) for one stage from a
# logs.tar.gz (one archive of <output_dir>/logs, made to cut inodes), so a rerun skips finished tasks.
# usage: restore_completions.sh <logs.tar.gz> <stage> <logging_dir>
#   e.g. restore_completions.sh .../data/tokenized/logs.tar.gz compact_stage1 /path/to/logs/compact_stage1
set -euo pipefail
[ $# -eq 3 ] || { sed -n '2,6p' "$0"; exit 1; }
A=$1; STAGE=$2; DEST=$3

tar -tzf "$A" "logs/$STAGE/executor.json" >/dev/null 2>&1 \
  || { echo "stage '$STAGE' not in $A; stages:"; tar -tzf "$A" | awk -F/ 'NF>=3 && $3=="" {print "  "$2}'; exit 1; }
if [ -d "$DEST/completions" ] && [ -n "$(ls -A "$DEST/completions")" ]; then
  echo "$DEST/completions already has markers; refusing to overwrite"; exit 1
fi

mkdir -p "$DEST"
tar -xzf "$A" -C "$DEST" --strip-components=2 \
  --wildcards "logs/$STAGE/completions/*" "logs/$STAGE/executor.json"

ws=$(grep -o '"world_size": *[0-9]*' "$DEST/executor.json" | head -1 | grep -o '[0-9]*$')
nc=$(ls -f "$DEST/completions" | grep -vc '^\.\.\?$' || true)
echo "restored $nc / $ws completion markers into $DEST/completions"
[ "$nc" = "$ws" ] || echo "WARNING: not all tasks were complete; a rerun will redo $((ws - nc)) tasks"
echo "Reminder: the rerun must use the same world_size/tasks ($ws) and logging_dir=$DEST."
echo "When done, re-tar or delete $DEST to keep inodes down."
