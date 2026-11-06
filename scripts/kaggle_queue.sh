#!/bin/bash
# usage: KAGGLE_API_TOKEN=... scripts/kaggle_queue.sh <owner> <dir1> <dir2> ...   (dirs under outputs/kaggle_kernels)
# Pushes kernels in order, keeping at most 2 running (Kaggle's batch GPU session limit). Exits when all are pushed,
# and STOPS as soon as any pushed kernel ends in ERROR/CANCEL (a free slot from a failure is not a reason to continue).
cd "$(dirname "$0")/.."
K=".venv/Scripts/uv.exe tool run --python 3.12 --with six kaggle"
owner=$1; shift; queue=("$@"); pushed=()
while [ ${#queue[@]} -gt 0 ]; do
  busy=0
  for d in "${pushed[@]}"; do
    st=$($K kernels status "$owner/polypgen-loco-$d" 2>&1 | tail -1)
    echo "$st" | grep -q "RUNNING\|QUEUED" && busy=$((busy+1))
    echo "$(date +%H:%M) $d: ${st##* }"
    if echo "$st" | grep -q "ERROR\|CANCEL"; then echo "!!! $d ended with ${st##* }: stopping the queue (read its log first)"; exit 2; fi
  done
  if [ $busy -lt 2 ]; then
    d=${queue[0]}; out=$($K kernels push -p "outputs/kaggle_kernels/$d" 2>&1 | tail -1); echo "$(date +%H:%M) push $d: $out"
    if echo "$out" | grep -q "successfully pushed"; then pushed+=("$d"); queue=("${queue[@]:1}"); sleep 60; continue
    elif ! echo "$out" | grep -q "Maximum batch GPU session"; then echo "!!! push failed, stopping"; exit 2; fi
  fi
  sleep 600
done
echo "all pushed: ${pushed[*]}"
