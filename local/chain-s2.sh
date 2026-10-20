#!/bin/bash
# After chain-s1 (launch it then): the thinking-off reference again (chain-s1's first run passed an empty argument
# to bench.py), the PCIe site's copies on their own stream with two banks of 16 slots (Xolotl's STRATA_D3_DMA_STREAM,
# off by default: the copies otherwise queue on the compute stream behind the kernels that read the slots), and
# the peer stores at the cards' limits (30 + 22 GiB).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-3gpu-ref "$B"
local/sweep.sh S-depth3 "$B" STRATA_DSPARK_DEPTH=3
local/sweep.sh S-dmastream "$B" STRATA_D3_DMA_STREAM=1 STRATA_D3_STAGE_SLOTS=16
local/sweep.sh S-peer3022 "$B" STRATA_PEER_GIB=30,22
# the stores filled from the ranked walk itself (the hottest non-VRAM experts; the arena then duplicates them
# until --arena-adapt swaps the duplicates out for file-tier experts)
local/sweep.sh S-fillany "$B" STRATA_PEER_FILL=any
# depth 2 (the best of chain-s1 so far) + speculative NVMe reads of layer l+2's predicted experts (STRATA_D3_SPEC,
# off by default): the file-tier reads are 57 blobs a pass after the 4K prompt (3.9 tok/s there)
local/sweep.sh S-d2spec8 "$B" STRATA_DSPARK_DEPTH=2 STRATA_D3_SPEC=8
local/sweep.sh S-d2spec16 "$B" STRATA_DSPARK_DEPTH=2 STRATA_D3_SPEC=16 STRATA_D3_SPEC_AHEAD=3
(cd /home/yuya/ds41 && python3 table.py S-3gpu-ref S-depth3 S-depth2 S-pcie07 S-pcie05 S-thr20 S-lpages S-dmastream S-peer3022 S-fillany S-d2spec8 S-d2spec16)
echo "$(date +%T) chain-s2 done"
