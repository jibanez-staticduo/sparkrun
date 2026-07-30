#!/bin/bash
# Non-destructive setup-readiness probe for `sparkrun setup check`.
# Emits key=value pairs on stdout; NEVER modifies host state. Diagnostic
# noise goes to stderr. Mirrors the parse style of spark_diagnose.sh.
#
# Params: {peers}  — space-separated peer hosts for the SSH-mesh probe.
set -uo pipefail

WHO=$(id -un 2>/dev/null || echo unknown)
echo "CHECK_USER=$WHO"

# --- Docker ---
if command -v docker >/dev/null 2>&1; then
    echo "CHECK_DOCKER_INSTALLED=1"
    if docker info >/dev/null 2>&1; then
        echo "CHECK_DOCKER_USABLE=1"
    else
        echo "CHECK_DOCKER_USABLE=0"
    fi
else
    echo "CHECK_DOCKER_INSTALLED=0"
    echo "CHECK_DOCKER_USABLE=0"
fi

if id -nG "$WHO" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
    echo "CHECK_DOCKER_GROUP=1"
else
    echo "CHECK_DOCKER_GROUP=0"
fi

# --- NVIDIA GPU / Container Toolkit / CDI ---
command -v nvidia-smi >/dev/null 2>&1 && echo "CHECK_GPU_PRESENT=1" || echo "CHECK_GPU_PRESENT=0"
command -v nvidia-ctk >/dev/null 2>&1 && echo "CHECK_NVIDIA_CTK=1" || echo "CHECK_NVIDIA_CTK=0"
if [ -s /etc/cdi/nvidia.yaml ]; then echo "CHECK_CDI_SPEC=1"; else echo "CHECK_CDI_SPEC=0"; fi

# --- earlyoom ---
command -v earlyoom >/dev/null 2>&1 && echo "CHECK_EARLYOOM_INSTALLED=1" || echo "CHECK_EARLYOOM_INSTALLED=0"
if systemctl is-active --quiet earlyoom 2>/dev/null; then
    echo "CHECK_EARLYOOM_ACTIVE=1"
else
    echo "CHECK_EARLYOOM_ACTIVE=0"
fi

# --- Sudoers entries (best-effort) ---
# Only inspect when passwordless sudo is available so the probe never blocks
# on a password prompt; otherwise report "unknown".
if sudo -n true 2>/dev/null; then
    if sudo -n test -e "/etc/sudoers.d/sparkrun-chown-$WHO" 2>/dev/null; then
        echo "CHECK_SUDOERS_CHOWN=1"
    else
        echo "CHECK_SUDOERS_CHOWN=0"
    fi
    if sudo -n test -e "/etc/sudoers.d/sparkrun-dropcaches-$WHO" 2>/dev/null; then
        echo "CHECK_SUDOERS_DROPCACHES=1"
    else
        echo "CHECK_SUDOERS_DROPCACHES=0"
    fi
else
    echo "CHECK_SUDOERS_CHOWN=unknown"
    echo "CHECK_SUDOERS_DROPCACHES=unknown"
fi

# --- SSH mesh (non-destructive) ---
# Attempt a BatchMode SSH to each peer; write no known_hosts entries.
# NOTE: `ssh -n` (stdin from /dev/null) is REQUIRED here. This whole script
# is delivered to the host via `ssh <host> bash -s`, so the script body IS the
# remote bash's stdin. Without -n the inner ssh would slurp the rest of that
# stdin (the remainder of this script), truncating execution so CHECK_COMPLETE
# never prints and the host is falsely reported unreachable.
PEERS="{peers}"
MESH_TOTAL=0
MESH_OK=0
for peer in $PEERS; do
    MESH_TOTAL=$((MESH_TOTAL + 1))
    if ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 "$peer" true 2>/dev/null; then
        MESH_OK=$((MESH_OK + 1))
    fi
done
echo "CHECK_MESH_TOTAL=$MESH_TOTAL"
echo "CHECK_MESH_OK=$MESH_OK"

echo "CHECK_COMPLETE=1"
