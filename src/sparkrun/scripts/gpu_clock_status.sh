#!/bin/bash
# Report per-GPU clock state plus any sparkrun boot-time clock unit.
# Read-only — no sudo required.
# Emits:
#   UNIT: none | <clockspec> <systemctl-enabled-state>
#   one CSV row per GPU: index, current SM MHz, max SM MHz, applications MHz, persistence
set -uo pipefail

UNIT_NAME=sparkrun-gpu-clock.service
UNIT_FILE=/etc/systemd/system/$UNIT_NAME

if [ -f "$UNIT_FILE" ]; then
    SPEC=$(grep -m1 -oE 'lock-gpu-clocks [^ ]+' "$UNIT_FILE" | awk '{print $2}')
    if [ -z "$SPEC" ]; then SPEC="?"; fi
    STATE=$(systemctl is-enabled "$UNIT_NAME" 2>/dev/null)
    if [ -z "$STATE" ]; then STATE="unknown"; fi
    echo "UNIT: $SPEC $STATE"
else
    echo "UNIT: none"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

nvidia-smi \
    --query-gpu=index,clocks.current.sm,clocks.max.sm,clocks.applications.graphics,persistence_mode \
    --format=csv,noheader,nounits
