#!/bin/bash
# How much of a few points is noise?
#
# Retraining one config after a checkpoint was lost, same data and
# hyperparameters, moved k16 from +83% to +38%. Everything in this project that
# turns on a few points needs to know its own spread before it can be reported.
#
# Three seeds through training, then the hybrid measured once per checkpoint, so
# the number that comes out is end-to-end: training variance and evaluation
# variance together, which is what a reader actually cares about.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/seeds

C="--no-lora --mem-depth --lr 5e-3 --k 32 --k-random 16,128 --steps 1000 --chunks 400 --eval-n 24 --dtype bf16 --model ./qwen3-0.6b"
for S in 0 1 2; do
  echo "=== train seed $S $(date +%H:%M:%S) ==="
  python -u train_memtok.py $C --seed $S > out/seeds/train_s$S.log 2>&1
  grep -E "^\[ 1000\]" out/seeds/train_s$S.log | tail -1
done

for S in 0 1 2; do
  echo "=== hybrid seed $S $(date +%H:%M:%S) ==="
  python -u hybrid.py --k 64 --anchors 512 --probe 16 --chunk 0 --hops 16 --n 12 \
    --ckpt ckpt/memtok_k32_s${S}_deep_frozen.pt > out/seeds/hyb_s$S.log 2>&1 || \
  python -u hybrid.py --k 64 --anchors 512 --probe 16 --chunk 0 --hops 16 --n 12 \
    --ckpt ckpt/memtok_k32_deep_s${S}_frozen.pt > out/seeds/hyb_s$S.log 2>&1
  grep -E "LAST SEG|NO state|ORACLE \+" out/seeds/hyb_s$S.log | head -3
done
echo "=== done $(date +%H:%M:%S) ==="
