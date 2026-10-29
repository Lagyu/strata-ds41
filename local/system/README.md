# local/system: this machine's system files

Copies of the files outside the repository that the LLM server relies on, at the paths they have on the machine
(Ubuntu 26.04, kernel 7.0, ROCm 7.14). Nothing installs them; they document the setup and make it rebuildable.

| Path | Purpose |
|---|---|
| `etc/systemd/system/strata-ds41.service` | The server: `local/run-ds41.sh` (API and web chat on 127.0.0.1:8150), then `local/warmup.sh`. `LimitMEMLOCK=infinity` for the pinned expert arena. Not enabled at boot: `sudo systemctl start strata-ds41`. |
| `home/yuya/dsh-work/tools/restart-ds41.sh` | Restart helper: stop, wait until every GPU has released its VRAM (a 95 GiB pinned arena takes a while to tear down), start, wait for the API. |
| `home/yuya/.config/strata-ds41/lan.env.example` | Optional LAN mode for `run-ds41.sh`: listen address and API key. The real file is private (mode 600). |
| `etc/default/grub.d/90-maya-iommu.cfg` | `iommu=pt`: with Ubuntu's default translated DMA the x16 RX 7900 XTX read pinned RAM at ~19 GB/s (Windows: ~22-27 GB/s). |
| `etc/default/grub.d/91-amdgpu-overdrive.cfg` | `amdgpu.ppfeaturemask` with the overdrive bit, which the undervolt needs. |
| `etc/amdgpu-undervolt.conf`, `usr/local/sbin/amdgpu-undervolt`, `etc/systemd/system/amdgpu-undervolt.service` | Both RX 7900 XTX: core clock cap 2800 MHz and 1040 mV (a -110 mV offset), applied at boot. |
| `usr/local/sbin/amdgpu-fans.sh`, `etc/systemd/system/amdgpu-fans.service` | GPU fan curves with a 50 % floor, for airflow over the M.2 drives that stream the experts. |
| `usr/local/sbin/fans-max.sh`, `etc/systemd/system/fans-max.service` | The board's fan headers at full speed (cools the OCuLink adapter). |
| `usr/local/sbin/cpu-power-cap.sh`, `etc/systemd/system/cpu-power-cap.service` | CPU package power cap (PL1 110 W / PL2 140 W). Disabled since 2026-10-21, after a CPU re-paste. |
| `etc/udev/rules.d/80-r9700-no-runpm.rules` | Keeps the R9700 out of runtime suspend: a failed wake from BACO (2026-10-21, then an OCuLink eGPU) left it unusable until a reboot. Matches the card's PCI id, so it follows the card to any slot. |

The `99-badram.cfg` memmap (one weak DRAM page) was dropped on 2026-10-26 with the module that had the weak cell.

To install on a similar machine (as root): copy the files to the same paths, then `systemctl daemon-reload`,
`update-grub` (the kernel command line takes effect at the next boot), `udevadm control --reload`, and
`systemctl enable` the tuning services you want. The paths under `/home/yuya` and `/mnt/win` in the configs are this
machine's; adjust them.
