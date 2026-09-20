#!/bin/bash
# Rented-GPU run plan. Billed by the hour, so this is meant to be started once
# and left alone -- no interactive debugging on the clock.
#
# Checks the card first: a 5090 is sm_120 and needs torch >= 2.7 / cu12.8, and
# an old image fails with "no kernel image available" only once training starts,
# which would waste the whole session.
set -u
cd "$(dirname "$0")"
LOG=${LOG:-./out/cloud}
mkdir -p "$LOG" ckpt out

echo "=== card check ==="
# Picks the dtype from what the card actually supports rather than from what it
# is called. A rented box may be Ampere (bf16 fine), Ada/Blackwell (fine), or
# Volta (fp16 only) -- and a vGPU slice does not advertise which. Guessing wrong
# either crashes on the first matmul or silently produces NaNs.
python - <<'PY' > .dtype_probe || { echo 'CARD CHECK FAILED -- stop, do not burn hours'; exit 1; }
import torch, sys
cc = torch.cuda.get_device_capability(0)
info = [f"torch {torch.__version__} | cuda {torch.version.cuda}",
        f"device {torch.cuda.get_device_name(0)} | sm_{cc[0]}{cc[1]}"]
(torch.zeros(8, device="cuda") + 1).sum().item()          # real kernel
dt = "fp16"
try:
    a = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    if torch.isfinite(a @ a).all() and torch.cuda.is_bf16_supported():
        dt = "bf16"
except Exception as e:
    info.append(f"bf16 unusable: {type(e).__name__}")
info.append(f"chosen dtype: {dt}")
print("\n".join(info), file=sys.stderr)
print(dt)
PY
cat .dtype_probe >/dev/null
DT=${DT:-$(cat .dtype_probe)}
echo "using dtype: $DT"
COMMON="--steps 1200 --chunks 800 --eval-n 64 --k 32 --k-random 16,128 --dtype $DT"

# 1) smoke test: five minutes, proves the whole path before committing hours
echo "=== smoke ==="
python -u train_memtok.py $COMMON --steps 40 --chunks 80 --eval-n 12 \
  --model ./qwen3-0.6b 2>&1 | tail -5 || exit 1

# 2) the scale axis -- the point of renting. Same config at every size, so the
#    numbers are actually comparable (locally 1.7B had to be crippled to fit).
for M in qwen3-0.6b qwen3-1.7b qwen3-4b; do
  [ -d "$M" ] || { echo "skip $M (not downloaded)"; continue; }
  echo "=== $M ==="
  python -u train_memtok.py $COMMON --model "./$M" > "$LOG/$M.log" 2>&1
  grep -E "^\[" "$LOG/$M.log" | tail -3
done

echo "=== all done ==="
grep -H -E "^\[ *1200\]" "$LOG"/*.log
