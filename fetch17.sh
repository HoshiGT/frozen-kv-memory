#!/bin/bash
# Same resume-until-complete loop as the 0.6B pull: the connection dies
# mid-transfer from here, so detect the stall and reconnect with byte ranges.
set -u
D=/home/hoshi/Claude/memzip/qwen3-1.7b
B=https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main
M=https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/main
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  total=$(curl -sIL "$B/$f" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  [ -z "$total" ] && { echo "no size for $f"; exit 1; }
  echo "$f -> $total bytes"
  for try in $(seq 1 300); do
    have=$(stat -c%s "$D/$f" 2>/dev/null || echo 0)
    [ "$have" -ge "$total" ] && { echo "  complete"; break; }
    url=$B; [ $((try % 2)) -eq 0 ] && url=$M
    curl -sL -C - --speed-limit 40000 --speed-time 15 --connect-timeout 15 \
         --max-time 300 -o "$D/$f" "$url/$f"
    sleep 1
  done
done
echo ALLDONE
