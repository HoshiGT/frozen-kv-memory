#!/bin/bash
# Training the recursion itself -- the one part of the architecture that no
# number so far has verified. Everything before this was a single-hop checkpoint
# extrapolating to multiple hops zero-shot.
#
# Two runs, because "does it need a single-hop warmup" is exactly the kind of
# question that got answered wrong today by assuming instead of measuring:
#   warm    -- start from the best single-hop checkpoint (+104% at 16:1)
#   scratch -- random slots, learns the recursion from nothing
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/recur ckpt

COMMON="--mem-depth --k 32 --hops 1,6 --bptt 2 --steps 600 --chunks 200 --eval-n 16 --lr 5e-3"

echo "=== warm start $(date +%H:%M:%S) ==="
python -u train_recur.py $COMMON --init ckpt/memtok_k32_deep_frozen.pt \
  > out/recur/warm.log 2>&1
mv out/recur_k32_h1-6_deep.json out/recur/warm.json 2>/dev/null
mv ckpt/recur_k32_h1-6_deep.pt ckpt/recur_warm.pt 2>/dev/null
grep -E "^\[" out/recur/warm.log | tail -2

echo "=== from scratch $(date +%H:%M:%S) ==="
python -u train_recur.py $COMMON > out/recur/scratch.log 2>&1
mv out/recur_k32_h1-6_deep.json out/recur/scratch.json 2>/dev/null
mv ckpt/recur_k32_h1-6_deep.pt ckpt/recur_scratch.pt 2>/dev/null
grep -E "^\[" out/recur/scratch.log | tail -2

echo "=== done $(date +%H:%M:%S) ==="
for f in out/recur/*.log; do
  printf '%-10s %s\n' "$(basename "$f" .log)" "$(grep -E '^\[' "$f" | tail -1)"
done
