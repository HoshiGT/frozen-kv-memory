#!/bin/bash
# Does the ~60% ceiling move with budget?
#
# Training could not shift it (0.6B and 1.7B both), and scale could not shift it
# (+60% vs +61%). Budget is the one lever never pulled. The checkpoint was
# trained with random budgets 16-128, so every k here is in-distribution -- this
# measures the same weights read at different widths, not four different models.
#
# Three outcomes, and the third is the one worth watching for:
#   linear        budget is the bottleneck, plan B works
#   sub-linear    something else also binds, diminishing returns
#   flat          the bottleneck is NOT capacity, and the story so far is wrong
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/budget

for K in 16 32 64 128; do
  echo "##### k=$K $(date +%H:%M:%S) #####"
  python -u train_recur.py --mem-depth --eval-only --k $K --eval-hops 1,2,4,8 \
    --chunks 120 --eval-n 24 --model ./qwen3-0.6b \
    --init ckpt/memtok_k32_deep_frozen.pt 2>&1 | grep -E "^\[init\]|hops="
done
echo "===== done $(date +%H:%M:%S) ====="
