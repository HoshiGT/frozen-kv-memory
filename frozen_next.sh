#!/bin/bash
# The lr sweep changed the picture, so re-test what was concluded under lr=1e-3.
#
#   frozen, lr 1e-3  -> -246%   (the lr every earlier run used)
#   frozen, lr 5e-3  ->  +66%
#   frozen, lr 2e-2  -> -124%
#
# Two things follow. First, "0.6B is too small to learn this" (-7.6% with LoRA,
# 1200 steps) may be nothing but a badly chosen lr. Second, a frozen backbone
# beating a trained one is worth checking at matched lr before believing it.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/frozen

BASE="--chunks 400 --eval-n 24 --k 32 --k-random 16,128 --dtype bf16 --model ./qwen3-0.6b"

run() {  # run <name> <extra args...>
  local name=$1; shift
  echo "=== $name $(date +%H:%M:%S) ==="
  python -u train_memtok.py $BASE "$@" > "out/frozen/$name.log" 2>&1
  for j in out/memtok_k32_frozen.json out/memtok_k32_deep_frozen.json out/memtok_k32.json; do
    [ -f "$j" ] && mv "$j" "out/frozen/$name.json"
  done
  grep -E "^\[" "out/frozen/$name.log" | tail -1
}

# 1) how far does the frozen setup actually go, given steps
run long1500 --no-lora --lr 5e-3 --steps 1500

# 2) the fair ablation: same lr, backbone allowed to adapt. If this loses to the
#    frozen run, LoRA is actively harmful at this scale rather than merely weak.
run lora5e-3 --lr 5e-3 --steps 400

# 3) is 5e-3 a peak or a plateau edge
run lr3e-3 --no-lora --lr 3e-3 --steps 400
run lr8e-3 --no-lora --lr 8e-3 --steps 400

# 4) per-layer bias for the memory slots: 0.9M params, still frozen backbone
run deep5e-3 --no-lora --mem-depth --lr 5e-3 --steps 400

echo "=== all done $(date +%H:%M:%S) ==="
for f in out/frozen/*.log; do
  printf '%-24s %s\n' "$(basename "$f" .log)" "$(grep -E '^\[ ' "$f" | tail -1)"
done
