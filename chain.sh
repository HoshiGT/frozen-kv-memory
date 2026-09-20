#!/bin/bash
# Unattended run chain for the rented box.
#
# All three sizes at the full 1200 steps. Hourly cost is a few yuan, so the
# budget is not worth trading against clean data -- every point gets the same
# config, which is exactly what the local runs could not afford.
#
# Each stage waits for what it needs (previous training, or the download that
# has not finished yet) so nothing has to be launched by hand.
set -u
cd "$(dirname "$0")"
P=/root/miniconda3/bin/python
COMMON="--chunks 800 --eval-n 64 --k 32 --k-random 16,128 --dtype bf16"

wait_for_training() { while pgrep -f "train_memtok" >/dev/null; do sleep 20; done; }
wait_for_model() {   # $1 = dir; complete when pget printed READY for it
  for i in $(seq 1 240); do
    grep -q "^$1 READY" p2.log 2>/dev/null && return 0
    sleep 15
  done
  return 1
}

echo "=== chain started $(date +%H:%M:%S) ==="

wait_for_training
echo "--- 0.6B finished $(date +%H:%M:%S) ---"
tail -2 run_0p6b.log

echo "=== 1.7B $(date +%H:%M:%S) ==="
$P -u train_memtok.py $COMMON --steps 1200 --model ./qwen3-1.7b > run_1p7b.log 2>&1
tail -2 run_1p7b.log

if wait_for_model qwen3-4b; then
  echo "=== 4B $(date +%H:%M:%S) ==="
  $P -u train_memtok.py $COMMON --steps 1200 --model ./qwen3-4b > run_4b.log 2>&1
  tail -2 run_4b.log
else
  echo "4B never finished downloading; skipped"
fi

echo "=== chain done $(date +%H:%M:%S) ==="
{
  echo "# scale axis"
  for m in 0p6b 1p7b 4b; do
    f=run_$m.log
    [ -f "$f" ] && echo "## $m" && grep -E "^\[" "$f" | grep -v transformers | tail -3
  done
} > SUMMARY.txt
cat SUMMARY.txt
