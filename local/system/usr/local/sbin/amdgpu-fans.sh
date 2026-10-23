#!/bin/bash
# amdgpu-fans.sh - a higher fan curve on every AMD card (hotspot -> fan %), the zero-RPM mode off and a 50% floor, so
# the cards' fans keep air moving over the M.2 drives even when the GPUs are cool (the P41 that streams the experts
# sat at 82-83 C, its throttle region, during the 2026-10-20 sweeps).  Applied at boot by amdgpu-fans.service;
# re-run after a driver reset.  Undo: write "r" to each gpu_od/fan_ctrl file (the card's defaults return).
set -u
MIN=${AMDGPU_FAN_MIN:-50}
for d in /sys/class/drm/card?/device; do
  f=$d/gpu_od/fan_ctrl
  [ -d "$f" ] || continue
  slot=$(grep PCI_SLOT_NAME "$d/uevent" | cut -d= -f2)
  # zero-RPM off (the R9700 has no such mode: its file is empty and the write fails, harmlessly), and the zero-RPM
  # stop temperature at its minimum: with it at the default 55 C the chipset XTX's fan stayed at 0 rpm at 46 C
  if [ -s "$f/fan_zero_rpm_enable" ]; then
    echo 0 > "$f/fan_zero_rpm_enable" 2>/dev/null && echo c > "$f/fan_zero_rpm_enable" 2>/dev/null
  fi
  if [ -s "$f/fan_zero_rpm_stop_temperature" ]; then
    echo 25 > "$f/fan_zero_rpm_stop_temperature" && echo c > "$f/fan_zero_rpm_stop_temperature"
  fi
  # the floor first: the curve's lowest point may not go below it
  echo "$MIN" > "$f/fan_minimum_pwm" && echo c > "$f/fan_minimum_pwm"
  echo "0 40 $MIN" > "$f/fan_curve"
  echo "1 55 65" > "$f/fan_curve"
  echo "2 65 80" > "$f/fan_curve"
  echo "3 75 90" > "$f/fan_curve"
  echo "4 85 100" > "$f/fan_curve"
  echo c > "$f/fan_curve"
  echo "amdgpu-fans: $slot: zero-RPM off (stop temp 25 C), floor ${MIN}%, curve 40C ${MIN}% / 55C 65% / 65C 80% / 75C 90% / 85C 100%"
done
# The chipset-slot XTX (0000:0c:00.0) ignores the floor while it idles (its fan stayed at 0-360 rpm at 46 C with the
# floor at 50%, and manual pwm1 is refused with EINVAL): its fan target temperature goes to the minimum (25 C), which
# makes the controller run the fan (measured: 1055 rpm, 64%, at 46 C).  It only ever runs the peer expert tier, and
# its fan is the one over the lower M.2 slot.
for d in /sys/class/drm/card?/device; do
  [ "$(grep PCI_SLOT_NAME "$d/uevent" 2>/dev/null | cut -d= -f2)" = "0000:0c:00.0" ] || continue
  f=$d/gpu_od/fan_ctrl
  echo 25 > "$f/fan_target_temperature" && echo c > "$f/fan_target_temperature" && echo "amdgpu-fans: 0000:0c:00.0: fan target temperature 25 C"
done
