#!/bin/bash
# Lock or reset the GPU SM clock ceiling via nvidia-smi, and optionally manage a
# systemd oneshot that reapplies the lock at boot (the driver forgets it).
# Runs as root via run_remote_sudo_script (no sudo prefix needed).
# Params: clock_args (nvidia-smi clock flag), result_label (status line),
#         unit_action (install|remove|keep)
set -euo pipefail

UNIT_NAME=sparkrun-gpu-clock.service
UNIT_FILE=/etc/systemd/system/$UNIT_NAME

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi
NVIDIA_SMI=$(command -v nvidia-smi)

nvidia-smi {clock_args} >/dev/null
echo "{result_label}"

case "{unit_action}" in
install)
    tee "$UNIT_FILE" > /dev/null << UNIT_EOF
[Unit]
# Written by: sparkrun setup throttle-gpu-clock --persistent
Description=sparkrun GPU clock lock
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$NVIDIA_SMI {clock_args}

[Install]
WantedBy=multi-user.target
UNIT_EOF
    systemctl daemon-reload
    systemctl enable "$UNIT_NAME" >/dev/null 2>&1
    echo "UNIT: installed, enabled at boot"
    ;;
remove)
    if [ -f "$UNIT_FILE" ]; then
        systemctl disable "$UNIT_NAME" >/dev/null 2>&1 || true
        rm -f "$UNIT_FILE"
        systemctl daemon-reload
        echo "UNIT: removed"
    else
        echo "UNIT: none"
    fi
    ;;
*)
    if [ -f "$UNIT_FILE" ]; then
        echo "UNIT: present and unchanged ($(grep -m1 -oE 'lock-gpu-clocks [^ ]+' "$UNIT_FILE" || true))"
    fi
    ;;
esac

echo "CURRENT: $(nvidia-smi --query-gpu=clocks.sm,persistence_mode --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
