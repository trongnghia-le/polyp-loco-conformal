#!/bin/bash
# usage: scripts/push_when_free.sh <dir-to-push> <busy-slug-1> <busy-slug-2> [poll-seconds]
# Kaggle allows 2 concurrent batch GPU sessions: wait until one of the two kernels is no longer RUNNING/QUEUED, then push.
cd "$(dirname "$0")/.."
K=".venv/Scripts/uv.exe tool run --python 3.12 --with six kaggle"
dir=$1; a=$2; b=$3; every=${4:-600}
while true; do
  sa=$($K kernels status "$a" 2>&1 | tail -1); sb=$($K kernels status "$b" 2>&1 | tail -1)
  echo "$(date +%H:%M) $a: ${sa##* } | $b: ${sb##* }"
  if ! echo "$sa" | grep -q "RUNNING\|QUEUED" || ! echo "$sb" | grep -q "RUNNING\|QUEUED"; then
    if echo "$sa$sb" | grep -q "RUNNING\|QUEUED\|COMPLETE\|ERROR"; then   # status calls really answered (not a network error)
      $K kernels push -p "$dir" 2>&1 | tail -2; exit 0
    fi
  fi
  sleep "$every"
done
