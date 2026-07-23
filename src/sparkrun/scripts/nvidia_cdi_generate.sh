#!/bin/bash
# Generate the NVIDIA CDI (Container Device Interface) spec so Docker can
# resolve the `--device nvidia.com/gpu=...` flags the docker executor emits.
# Some DGX Spark hosts need this generated explicitly for full CDI support.
# Uses sudo -n (non-interactive).
set -uo pipefail

OUTPUT="/etc/cdi/nvidia.yaml"

# Self-skip where the NVIDIA Container Toolkit isn't installed (non-NVIDIA
# hosts, or hosts where the toolkit hasn't been set up yet) — never fatal.
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "SKIPPED: nvidia-ctk not found (NVIDIA Container Toolkit not installed)"
    exit 0
fi

# nvidia-ctk writes $OUTPUT itself (as root under sudo), and its own logs go to
# stderr — deliberately no shell redirect to a temp file: a fixed /tmp path
# collides across users/runs and, under fs.protected_regular, even root can't
# write another user's file in /tmp.
if ! sudo -n mkdir -p /etc/cdi; then
    echo "ERROR: could not create /etc/cdi (sudo required)"
    exit 1
fi
if ! sudo -n nvidia-ctk cdi generate --output="$OUTPUT"; then
    echo "ERROR: nvidia-ctk cdi generate failed"
    exit 1
fi

if [ ! -s "$OUTPUT" ]; then
    echo "ERROR: $OUTPUT was not created"
    exit 1
fi

DEVICES=$(nvidia-ctk cdi list 2>/dev/null | grep -c 'nvidia.com/gpu=' || true)
if [ -n "$DEVICES" ] && [ "$DEVICES" != "0" ]; then
    echo "GENERATED: $OUTPUT ($DEVICES device(s))"
else
    echo "GENERATED: $OUTPUT"
fi
