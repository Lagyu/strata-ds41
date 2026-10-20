#!/bin/bash
# chain-d1: DeepInfer's decoder SWA bounded replay (--enc-bounded --enc-replay 140) - speed on the serve path (bench + turnbench),
# then the next-token quality: the last prompt row's logits of the 8 held-out sequences, exact replay (2560) vs bounded (140).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-bnd140 "$B" --enc-bounded=on --enc-replay=140
grep -E "enc-skip: (BOUNDED|first batch)" local/serve-engine.log | tail -2 | cut -c1-160
python3 tools/turnbench.py http://127.0.0.1:8150 --base 4000 --deltas 64,182,512 --log local/serve-engine.log 2>&1 | grep -E "^turn" | cut -c1-150
(cd /home/yuya/ds41 && python3 table.py S-cap8 S-cap16 S-bnd140)
sudo systemctl stop strata-ds41.service; sleep 20
E=local/eval-lucebox; O=$E/bounded; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib
for i in 0 1 2 3 4 5 6 7; do
  for mode in exact bnd; do
    extra=""; [ $mode = bnd ] && extra="--enc-bounded --enc-replay 140"
    env HIP_VISIBLE_DEVICES=0 STRATA_PF_NO_DUMMY=1 STRATA_NO_LARGEPAGES=1 ./build-hip/ds41_generate \
      -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file "$E/base-3gpu/seq$i.i32" -n 1 --temp 0 \
      --backend cuda --experts gpu --arena-gib 75 --ctx 4096 --kv-fp4 --threads 12 --prompt-batch 2048 \
      --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin \
      --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
      --route-bias 0 --route-bias-file 0 --dump-last "$O/seq$i.$mode.f32" $extra > "$O/seq$i.$mode.out" 2> "$O/seq$i.$mode.log"
    grep -E "^prompt: |enc-skip: (BOUNDED|first)" "$O/seq$i.$mode.log" | tail -2 | sed "s/^/seq$i $mode: /" | cut -c1-160
  done
  /home/yuya/llm-src/hfenv/bin/python local/kl-last.py "$O/seq$i.exact.f32" "$O/seq$i.bnd.f32" | sed "s/^/seq$i: /"
done
echo "$B" > local/active-config
sudo systemctl start strata-ds41.service
echo "$(date +%T) chain-d1 done"
