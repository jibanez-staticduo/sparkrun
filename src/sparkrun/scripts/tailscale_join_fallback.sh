#!/bin/bash
# Join this host to a tailnet. Fallback variant: the whole script is executed
# under `sudo -S` (root), so privileged commands run bare. Used when the primary
# variant's non-interactive `sudo -n` failed and a sudo password is available.
# Placeholders are filled via str.format(), so this file must contain no literal
# curly braces (use unbraced $VAR references only).
set -uo pipefail

AUTHKEY='{authkey}'
TAGS='{tags}'
HOSTNAME_ARG='{hostname_arg}'
EXTRA_ARGS='{extra_args}'

if command -v tailscale >/dev/null 2>&1; then
    echo "TS_INSTALL=present"
else
    echo "TS_INSTALL=installing"
    if ! curl -fsSL https://tailscale.com/install.sh | sh >/dev/null 2>&1; then
        echo "TS_ERROR=install_failed"
        exit 1
    fi
fi

# Ensure the tailscaled daemon is running before `tailscale up` (see the primary
# variant for the rationale). Runs bare — the whole script is already under sudo.
if tailscale status >/dev/null 2>&1; then
    echo "TS_DAEMON=running"
elif [ -d /run/systemd/system ]; then
    systemctl enable --now tailscaled >/dev/null 2>&1 || true
    echo "TS_DAEMON=systemd"
else
    if [ -c /dev/net/tun ]; then
        TS_TUN_ARG=
    else
        TS_TUN_ARG=--tun=userspace-networking
    fi
    mkdir -p /var/lib/tailscale /var/run/tailscale >/dev/null 2>&1 || true
    nohup tailscaled $TS_TUN_ARG --state=/var/lib/tailscale/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock >/tmp/sparkrun-tailscaled.log 2>&1 </dev/null &
    echo "TS_DAEMON=manual"
    n=0
    while [ $n -lt 15 ]; do
        if [ -S /var/run/tailscale/tailscaled.sock ]; then break; fi
        n=$((n + 1))
        sleep 1
    done
fi

if ! tailscale up --authkey="$AUTHKEY" --advertise-tags="$TAGS" $HOSTNAME_ARG $EXTRA_ARGS 2>&1; then
    echo "TS_ERROR=up_failed"
    exit 1
fi

TS_IP=$(tailscale ip -4 2>/dev/null | head -n1)
echo "TS_IP=$TS_IP"
echo "TS_OK=1"
