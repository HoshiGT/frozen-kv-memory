#!/bin/bash
# Frozen backbone: can a compression policy be learned from the input side alone?
#
# Everything positive so far trained mem.emb *and* LoRA together, so the reader
# adapted to the writer. This freezes the reader completely -- the only trainable
# thing in the system is k vectors, 32k-131k parameters against a 0.6B model that
# never changes. If that works at all it is the deployable form: original weights
# untouched, memory as a plug-in.
#
# Prompt-tuning-scale parameter counts usually want a much larger lr than the
# 1e-3 used when LoRA was carrying most of the capacity, so sweep that first --
# locally this costs minutes, not money.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out ckpt out/frozen

COMMON="--no-lora --chunks 400 --eval-n 24 --k 32 --k-random 16,128 --dtype bf16 --model ./qwen3-0.6b"

for LR in 1e-3 5e-3 2e-2; do
  echo "=== lr=$LR $(date +%H:%M:%S) ==="
  python -u train_memtok.py $COMMON --steps 400 --lr $LR \
    > out/frozen/lr$LR.log 2>&1
  mv out/memtok_k32_frozen.json "out/frozen/lr$LR.json" 2>/dev/null
  grep -E "^\[" "out/frozen/lr$LR.log" | tail -2
done

echo "=== sweep done $(date +%H:%M:%S) ==="
for LR in 1e-3 5e-3 2e-2; do
  echo "lr=$LR: $(grep -E '^\[ ' out/frozen/lr$LR.log | tail -1)"
done
