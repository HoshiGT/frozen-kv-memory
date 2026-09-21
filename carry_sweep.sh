#!/bin/bash
# Budget is not the bottleneck; re-encoding is. So stop re-encoding everything.
#
# The state stays exactly k wide. What changes is how many of its slots get
# rewritten each hop -- a slot that is never rewritten cannot accumulate the
# loss that a rewritten one does.
#
#   keep=0            today: every slot re-encoded every hop
#   keep-which=old    earliest slots stay resident, like an attention sink
#   keep-which=new    FIFO window, oldest evicted
#
# Zero-shot: the checkpoint was trained with full rewriting, so any gain here is
# a lower bound on what training this would give.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
K=64
for SPEC in "0 old" "16 old" "32 old" "48 old" "16 new" "32 new" "48 new"; do
  set -- $SPEC
  printf '%-18s ' "keep=$1 ($2)"
  python -u train_recur.py --mem-depth --eval-only --k $K --keep $1 --keep-which $2 \
    --eval-hops 1,2,4,8 --chunks 120 --eval-n 24 --model ./qwen3-0.6b \
    --init ckpt/memtok_k32_deep_frozen.pt 2>&1 | grep -E "^\[init\]" | sed 's/\[init\] | //'
done
