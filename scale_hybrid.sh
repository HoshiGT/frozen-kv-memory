#!/bin/bash
# The hybrid architecture has only been measured on 0.6B. Three questions:
#
#   does +86.8% hold, improve or decay with scale
#   does the surprisal/speculative ordering flip (it already flipped once, at
#     2048 tokens, where surprisal led)
#   is the abstract state still saturated at 64 slots on a bigger model
#
# 1.7B needs --pred-len 192 on an 8GB card: the tail's logits alone are 311MB at
# 512 tokens over a 151k vocabulary.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/scale

echo "=== 1.7B, 4096 tokens, 7:1 $(date +%H:%M:%S) ==="
python -u hybrid.py --model ./qwen3-1.7b --ckpt ckpt/memtok_17b_deep.pt \
  --k 64 --anchors 512 --probe 128 --chunk 0 --hops 8 --n 12 \
  > out/scale/hybrid17b.log 2>&1
grep -E "budget|abstract \(accum|SURPRISAL \+|SPECULATIVE|ORACLE \+|single-topic|multi-topic" \
  out/scale/hybrid17b.log

echo "=== 0.6B same settings for comparison $(date +%H:%M:%S) ==="
python -u hybrid.py --k 64 --anchors 512 --probe 128 --chunk 0 --hops 8 --n 12 \
  > out/scale/hybrid06b.log 2>&1
grep -E "abstract \(accum|SURPRISAL \+|SPECULATIVE|ORACLE \+|single-topic|multi-topic" \
  out/scale/hybrid06b.log

echo "=== done $(date +%H:%M:%S) ==="
