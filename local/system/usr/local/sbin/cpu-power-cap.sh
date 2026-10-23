#!/bin/bash
# i7-14700K package power cap (RAPL PL1 110 W / PL2 140 W): with the CPU lane's threads and the
# SSD read helpers the package hit 99 C at the stock 125/230 W on 2026-10-19 (fans already at
# full speed). The lane is memory-bound, so the cap costs it little. Revert: 125000000 / 230000000.
R=/sys/class/powercap/intel-rapl:0
[ -d "$R" ] || exit 0
echo 110000000 > "$R/constraint_0_power_limit_uw"
echo 140000000 > "$R/constraint_1_power_limit_uw"
echo "cpu-power-cap: PL1=$(cat $R/constraint_0_power_limit_uw) PL2=$(cat $R/constraint_1_power_limit_uw)"
