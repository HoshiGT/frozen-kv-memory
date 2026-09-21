#!/bin/bash
# Third scale point, locally. 0.6B and 1.7B agreed on every core number, so 4B
# is about whether that keeps holding rather than about finding something new.
#
# 4B in bf16 is 8GB of weights alone, which does not fit beside its own
# activations on this card, so the backbone is NF4-quantised. That is sound for
# what we do -- the backbone is frozen, and upper/lower/mem are all measured on
# the same quantised model, so quantisation error largely cancels in the ratio.
# --load-4bit has never actually been run, hence the smoke test first.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out/4b

echo "=== waiting for download $(date +%H:%M:%S) ==="
# pget.sh's ranged chunks come out wrong against hf-mirror's redirects -- one
# shard downloaded at 6417MB against an index total of 7.49GB for all three.
# ModelScope's own SDK is slower here but correct.
until grep -q "^DONE" /tmp/ms4b.log 2>/dev/null; do
  if ! pgrep -f snapshot_download >/dev/null; then echo "download died"; exit 1; fi
  sleep 30
done
du -sh qwen3-4b

echo "=== smoke: does --load-4bit work at all $(date +%H:%M:%S) ==="
if ! python -u train_memtok.py --no-lora --mem-depth --load-4bit --lr 5e-3 \
      --k 32 --k-random 16,128 --pred-len 192 --steps 20 --chunks 60 --eval-n 6 \
      --model ./qwen3-4b > out/4b/smoke.log 2>&1; then
  echo "FAILED:"; grep -iE "error|memory" out/4b/smoke.log | tail -3; exit 1
fi
grep -E "^\[" out/4b/smoke.log | tail -2

echo "=== train $(date +%H:%M:%S) ==="
python -u train_memtok.py --no-lora --mem-depth --load-4bit --lr 5e-3 \
  --k 32 --k-random 16,128 --pred-len 192 --steps 1000 --chunks 400 --eval-n 24 \
  --model ./qwen3-4b > out/4b/train.log 2>&1
grep -E "^\[" out/4b/train.log | tail -2
mv ckpt/memtok_k32_deep_nf4_frozen.pt ckpt/memtok_4b_deep.pt 2>/dev/null || \
  mv ckpt/memtok_k32_deep_frozen_nf4.pt ckpt/memtok_4b_deep.pt 2>/dev/null || true
ls ckpt/ | grep -i 4b

echo "=== done $(date +%H:%M:%S) ==="
