#!/bin/bash
# Pull the models on the rented box (its link is far faster than uploading from
# home). Same resume loop as locally: HF stalls mid-transfer and needs ranged
# reconnects rather than a dead socket.
set -u
cd "$(dirname "$0")"
for spec in "Qwen3-0.6B:qwen3-0.6b" "Qwen3-1.7B:qwen3-1.7b" "Qwen3-4B:qwen3-4b"; do
  name=${spec%%:*}; dir=${spec##*:}
  mkdir -p "$dir"; cd "$dir"
  B=https://huggingface.co/Qwen/$name/resolve/main
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
           vocab.json merges.txt model.safetensors.index.json; do
    [ -f "$f" ] || curl -sfL -o "$f" "$B/$f" 2>/dev/null
  done
  if [ -f model.safetensors.index.json ]; then
    shards=$(python -c "import json;print(' '.join(sorted(set(json.load(open('model.safetensors.index.json'))['weight_map'].values()))))")
  else
    shards=model.safetensors
  fi
  for f in $shards; do
    total=$(curl -sIL "$B/$f" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
    for try in $(seq 1 200); do
      have=$(stat -c%s "$f" 2>/dev/null || echo 0)
      [ -n "$total" ] && [ "$have" -ge "$total" ] && break
      curl -sL -C - --speed-limit 40000 --speed-time 15 --max-time 600 -o "$f" "$B/$f"
    done
    echo "  $name/$f $(stat -c%s "$f" 2>/dev/null)"
  done
  cd ..
done
echo MODELS READY
