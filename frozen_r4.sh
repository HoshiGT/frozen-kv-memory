#!/bin/bash
# Round 4. Two questions worth the GPU time:
#
#   1. does the deep frozen prefix hold up at 1.7B? That is what decides whether
#      35B A3B is worth renting a 24G card for, and 1.7B weights are already here
#   2. does a compressed state survive being re-compressed? recur.py has existed
#      since this morning and has never run, because nothing was good enough to
#      recurse. deep400 (+90% at 16:1) is.
#
# The 1.7B run is smoke-tested first: the earlier OOM on this 8GB card came from
# LoRA's MLP adapters, which a frozen backbone does not have, but that is a
# prediction and predictions here have a poor record.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/frozen

D="--no-lora --mem-depth --lr 5e-3 --dtype bf16 --chunks 400 --eval-n 24 --k 32 --k-random 16,128"

echo "=== 1.7B smoke $(date +%H:%M:%S) ==="
if python -u train_memtok.py $D --steps 30 --chunks 60 --eval-n 8 \
     --model ./qwen3-1.7b > out/frozen/smoke17.log 2>&1; then
  echo "  fits; running 400 steps"
  python -u train_memtok.py $D --steps 400 --model ./qwen3-1.7b \
    > out/frozen/deep17b.log 2>&1
  mv out/memtok_k32_deep_frozen.json out/frozen/deep17b.json 2>/dev/null
  mv ckpt/memtok_k32_deep_frozen.pt ckpt/memtok_17b_deep.pt 2>/dev/null
  grep -E "^\[" out/frozen/deep17b.log | tail -1
else
  echo "  1.7B does not fit even frozen:"
  grep -iE "error|memory" out/frozen/smoke17.log | tail -3
fi

# recursion, on a pinned copy of the 0.6B deep checkpoint: every run writes the
# same tag-derived filename, so the later runs here would clobber it
echo "=== recursion $(date +%H:%M:%S) ==="
for H in 2 4 8; do
  python -u recur.py --k 32 --hops $H --mem-depth --model ./qwen3-0.6b \
    --ckpt ckpt/memtok_06b_deep400.pt > "out/frozen/recur_h$H.log" 2>&1
  echo "--- hops=$H ---"; tail -6 "out/frozen/recur_h$H.log"
done

# how far the deep prefix goes with more steps, at 0.6B
echo "=== deep 1000 $(date +%H:%M:%S) ==="
python -u train_memtok.py $D --steps 1000 --model ./qwen3-0.6b \
  > out/frozen/deep1000.log 2>&1
mv out/memtok_k32_deep_frozen.json out/frozen/deep1000.json 2>/dev/null
grep -E "^\[" out/frozen/deep1000.log | tail -1

echo "=== done $(date +%H:%M:%S) ==="
