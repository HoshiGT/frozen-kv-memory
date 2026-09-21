#!/bin/bash
# Does the abstract state get better when it knows anchors will be present?
#
# The control matters more than the treatment here: 400 more steps of anything
# might help, so the same schedule runs with anchors absent during training and
# present at evaluation. Only the difference between the two is evidence about
# division of labour.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
mkdir -p out/hybrid
C="--mem-depth --k 64 --anchors 512 --hops 2,4 --eval-hops 4,8 --steps 400 --chunks 160 --eval-n 16 --lr 2e-3"

echo "=== with anchors in training $(date +%H:%M:%S) ==="
python -u train_hybrid.py $C > out/hybrid/with.log 2>&1
grep -E "^\[" out/hybrid/with.log | tail -2

echo "=== control: no anchors in training $(date +%H:%M:%S) ==="
python -u train_hybrid.py $C --no-anchors-in-training > out/hybrid/without.log 2>&1
grep -E "^\[" out/hybrid/without.log | tail -2

echo "=== done $(date +%H:%M:%S) ==="
