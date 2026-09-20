#!/bin/bash
# Were the two negative results a property of the method, or of a 0.6B model?
#
# Hoshi's question, and a fair one: 1.7B already moved k=16 from -172% to +86%,
# so capacity conclusions demonstrably shift with scale. Both negatives rest on
# capacity or on representation quality, which is exactly what a small model is
# short of.
#
#   recursion  -- does the 8-hop number still sit near 60%, and does training it
#                 still gain nothing, when the model has room
#   budget     -- do the cheap signals predict hunger once hidden states are
#                 worth reading
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/scale

echo "=== 1.7B recursion, zero-shot + 150 steps $(date +%H:%M:%S) ==="
python -u train_recur.py --mem-depth --k 32 --hops 1,6 --bptt 2 --steps 150 \
  --chunks 120 --eval-n 12 --lr 5e-3 --model ./qwen3-1.7b \
  --init ckpt/memtok_17b_deep.pt > out/scale/recur17b.log 2>&1
grep -E "^\[" out/scale/recur17b.log | tail -4

echo "=== 1.7B budget signals $(date +%H:%M:%S) ==="
python -u topic.py --n 36 --model ./qwen3-1.7b \
  --ckpt ckpt/memtok_17b_deep.pt > out/scale/topic17b.log 2>&1
grep -E "Spearman|spread :|hunger :|predictable|surprising" out/scale/topic17b.log

echo "=== done $(date +%H:%M:%S) ==="
