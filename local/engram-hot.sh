#!/bin/bash
# engram-hot.sh - build the Engram hot-set pack (upstream's ds41_engram_hot, merged 2026-10-21): corpus -> rank ->
# simulate sizes on held-out text -> pack the top 4 GiB of rows from our Engram tables (the ROCMFP2S GGUF).
# CPU + disk only.  Log: logs/engram-hot.log.  Output: /home/yuya/models/dsv41-1017-strata/engram-hot-4g.bin
# (then --engram-hot PACK / STRATA_ENGRAM_HOT=PACK in a config; measured as its own sweep row, never silently).
set -u
R=/home/yuya/llm-src/strata-ds41
D=/home/yuya/models/dsv41-1017-strata
G=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
ENG=/home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf
PILE_DIR=/home/yuya/datasets/pile-10k
PILE=$PILE_DIR/train-00000-of-00001-4746b8785c874cc7.parquet
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$PILE_DIR" "$D/engram-corpus"
echo "== pile-10k $(date +%H:%M)"
[ -s "$PILE" ] || curl -sS -L --retry 3 -o "$PILE" "https://huggingface.co/datasets/NeelNanda/pile-10k/resolve/main/data/train-00000-of-00001-4746b8785c874cc7.parquet"
ls -la "$PILE"
echo "== corpus $(date +%H:%M)"
/home/yuya/llm-src/hfenv/bin/python3 -I "$R/local/engram-corpus-local.py" --out "$D/engram-corpus" --pile "$PILE" || exit 1
echo "== rank $(date +%H:%M)"
"$R/build-hip/ds41_engram_hot" rank --gguf "$G" --corpus "$D/engram-corpus/train.i32" --out "$D/engram-corpus/rank.bin" || exit 1
echo "== eval (held-out) $(date +%H:%M)"
"$R/build-hip/ds41_engram_hot" eval --gguf "$G" --rank "$D/engram-corpus/rank.bin" --corpus "$D/engram-corpus/heldout.i32" --gib 1,2,4,8 2>&1 | tail -12
echo "== pack 4 GiB $(date +%H:%M)"
"$R/build-hip/ds41_engram_hot" pack --gguf "$G" --dir "$ENG" --rank "$D/engram-corpus/rank.bin" --gib 4 --out "$D/engram-hot-4g.bin" --threads 32 || exit 1
ls -la "$D/engram-hot-4g.bin"
echo "engram-hot done $(date +%H:%M)"
