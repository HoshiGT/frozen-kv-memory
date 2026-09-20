#!/bin/bash
# Sweep the slot budget. Small m is the point: it removes the option of simply
# picking good tokens, so if superposition is reachable it has to show up here.
cd /home/hoshi/Claude/memzip
source ~/ctfenv/bin/activate
for m in 64 32 16 12 8 6; do
  echo "===== m=$m ====="
  python train.py --steps 400 --chunks 400 --eval-n 32 --m $m --sink 4 2>&1 \
    | grep -viE "warn|Loading weights|Token indices" \
    | grep -E "^\[|recent\(|compressor params"
done
