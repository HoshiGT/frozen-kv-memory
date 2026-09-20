#!/bin/bash
# Round 3, following what round 2 showed:
#
#   - frozen beats LoRA (+66% vs +8%), and LoRA got there by improving the lower
#     bound rather than by compressing -- so frozen is the honest measurement
#   - lr peaks at 5e-3
#   - 1500 steps moved capacity from k32 (+66 -> +11) to k72/k128 (+71 -> +89):
#     under a random budget the easy large-k samples dominate the gradient
#
# So: is the long-run degradation about the budget distribution or about the lr
# schedule? Two ways to separate that -- specialise on one budget, or keep the
# budget random but stop earlier.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/frozen

BASE="--chunks 400 --eval-n 24 --dtype bf16 --model ./qwen3-0.6b --no-lora --lr 5e-3"

run() {
  local name=$1; shift
  echo "=== $name $(date +%H:%M:%S) ==="
  python -u train_memtok.py $BASE "$@" > "out/frozen/$name.log" 2>&1
  for j in out/memtok_k32_frozen.json out/memtok_k32_deep_frozen.json; do
    [ -f "$j" ] && mv "$j" "out/frozen/$name.json"
  done
  grep -E "^\[" "out/frozen/$name.log" | tail -1
}

# the deep prefix, now that it can find the layers
run deep400 --mem-depth --k 32 --k-random 16,128 --steps 400

# one budget only: does 16:1 go further when nothing easier competes for capacity
run fixed32 --k 32 --steps 400

# same, longer -- if fixed-budget keeps improving, the 1500-step collapse was the
# budget distribution, not the schedule
run fixed32_1000 --k 32 --steps 1000

# the hard end, specialised
run fixed16 --k 16 --steps 400

echo "=== done $(date +%H:%M:%S) ==="
for f in out/frozen/*.log; do
  printf '%-18s %s\n' "$(basename "$f" .log)" "$(grep -E '^\[ ' "$f" | tail -1)"
done
