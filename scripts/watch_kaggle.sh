#!/bin/bash
# usage: scripts/watch_kaggle.sh <kernel-slug> [poll-seconds]   (needs ~/.kaggle/access_token)
# Exits when the kernel leaves RUNNING/QUEUED or its log shows a FAILED run; prints status + log tail.
cd "$(dirname "$0")/.."
K=".venv/Scripts/uv.exe tool run --python 3.12 --with six kaggle"
slug=$1; every=${2:-300}
while true; do
  st=$($K kernels status "$slug" 2>&1 | tail -1)
  log=$($K kernels logs "$slug" 2>&1)
  echo "$(date +%H:%M) $st | log lines: $(echo "$log" | wc -l)"
  if echo "$log" | grep -q "FAILED\|Traceback"; then echo "!!! problem in log"; echo "$log" | tail -25; exit 2; fi
  if ! echo "$st" | grep -q "RUNNING\|QUEUED"; then echo "$st"; echo "$log" | tail -40; exit 0; fi
  sleep "$every"
done
