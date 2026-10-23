#!/bin/bash
# Every fan header of the board's Nuvoton controller at full speed (mode 0): the CPU fan cools
# the OCuLink adapter in M2_1 (user's request, 2026-10-19). Reverting: write 5 (Smart Fan IV) back.
modprobe nct6775 2>/dev/null
for h in /sys/class/hwmon/hwmon*; do
  case "$(cat "$h/name" 2>/dev/null)" in nct*) ;; *) continue ;; esac
  for p in "$h"/pwm[1-7]_enable; do [ -f "$p" ] && echo 0 > "$p"; done
  echo "fans-max: $h set to full speed"
done
