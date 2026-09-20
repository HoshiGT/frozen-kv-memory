#!/bin/bash
# Resume-until-complete downloader. The HF connection dies mid-transfer from
# here, so detect the stall (--speed-limit/--speed-time) and reconnect with
# byte-range resume instead of waiting on a dead socket.
set -u
F=/home/hoshi/Claude/memzip/qwen3-0.6b/model.safetensors
TOTAL=1503300328
URLS=(
  "https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main/model.safetensors"
  "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors"
)
i=0
for try in $(seq 1 200); do
  have=$(stat -c%s "$F" 2>/dev/null || echo 0)
  [ "$have" -ge "$TOTAL" ] && { echo "COMPLETE $have"; exit 0; }
  url=${URLS[$((i % ${#URLS[@]}))]}; i=$((i+1))
  echo "try $try: $have/$TOTAL ($((have*100/TOTAL))%) via ${url:8:20}"
  curl -sL -C - --speed-limit 40000 --speed-time 15 --connect-timeout 15 \
       --max-time 300 -o "$F" "$url"
  sleep 1
done
echo "GAVE UP at $(stat -c%s "$F" 2>/dev/null)"
exit 1
