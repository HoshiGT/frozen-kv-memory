#!/bin/bash
# Parallel ranged download.
#
# A single connection through the academic proxy settles around 1.3 MB/s, which
# would put 13.5 GB of weights at three hours on an hourly-billed box. The cap
# is per-connection, not total, so N ranged requests in parallel multiply it.
# hf_transfer would do this too but ignores HTTP_PROXY and stalls at zero.
set -u
N=${N:-8}
source /etc/network_turbo >/dev/null 2>&1

pget() {
  local url=$1 out=$2
  local total
  total=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  [ -z "$total" ] && { echo "  !! no size for $out"; return 1; }
  local have
  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$have" -ge "$total" ]; then echo "  $out ok ($((total/1024/1024))MB)"; return 0; fi
  rm -f "$out" "$out".part*
  local chunk=$(( (total + N - 1) / N ))
  for i in $(seq 0 $((N-1))); do
    local s=$((i*chunk)) e=$(( (i+1)*chunk - 1 ))
    [ $e -ge $total ] && e=$((total-1))
    [ $s -gt $e ] && continue
    (
      for try in 1 2 3 4 5 6; do
        # -C - with -r resumes inside this part's own range
        curl -sL -r "$s-$e" -C - --speed-limit 30000 --speed-time 25 \
             --max-time 1800 -o "$out.part$i" "$url" && break
        sleep 2
      done
    ) &
  done
  wait
  cat "$out".part* > "$out" 2>/dev/null && rm -f "$out".part*
  local got
  got=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$got" -ge "$total" ]; then
    echo "  $out $((got/1024/1024))MB ok"
  else
    echo "  !! $out incomplete $((got/1024/1024))/$((total/1024/1024))MB"
    return 1
  fi
}

fetch_model() {
  local name=$1 dir=$2
  mkdir -p "$dir"; pushd "$dir" >/dev/null
  local B=https://huggingface.co/Qwen/$name/resolve/main
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
           vocab.json merges.txt model.safetensors.index.json; do
    [ -s "$f" ] || curl -sfL -o "$f" "$B/$f" 2>/dev/null
  done
  local shards
  if [ -s model.safetensors.index.json ]; then
    shards=$(/root/miniconda3/bin/python -c "import json;print(' '.join(sorted(set(json.load(open('model.safetensors.index.json'))['weight_map'].values()))))")
  else
    rm -f model.safetensors.index.json
    shards=model.safetensors
  fi
  for f in $shards; do pget "$B/$f" "$f" || true; done
  popd >/dev/null
  echo "$dir READY"
}

cd "$(dirname "$0")"
for spec in "$@"; do
  fetch_model "${spec%%:*}" "${spec##*:}"
done
echo ALLDONE
