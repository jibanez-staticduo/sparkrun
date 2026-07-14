#!/bin/bash
# Join this host to a tailnet. Primary variant: attempts privileged steps with
# non-interactive `sudo -n` so it succeeds silently on hosts with passwordless
# sudo. When sudo needs a password this script fails and the caller retries the
# _fallback variant under `sudo -S`. Placeholders are filled via str.format(),
# so this file must contain no literal curly braces (unbraced $VAR refs only).
set -uo pipefail

AUTHKEY='{authkey}'
TAGS='{tags}'
HOSTNAME_ARG='{hostname_arg}'
EXTRA_ARGS='{extra_args}'

if command -v tailscale >/dev/null 2>&1; then
    echo "TS_INSTALL=present"
else
    echo "TS_INSTALL=installing"
    if ! curl -fsSL https://tailscale.com/install.sh | sudo -n sh >/dev/null 2>&1; then
        echo "TS_ERROR=install_failed"
        exit 1
    fi
fi

# Ensure the tailscaled daemon is running before `tailscale up`:
#   - systemd hosts: start the service.
#   - containers without systemd (e.g. Thunder): start tailscaled manually; with
#     no /dev/net/tun it must run in userspace-networking mode.
# On a no-systemd host the daemon is unsupervised and won't survive a container
# restart (re-run join to bring it back). `tailscale up` is the authoritative
# check, so a start failure here is left for it to report.
if tailscale status >/dev/null 2>&1; then
    echo "TS_DAEMON=running"
elif [ -d /run/systemd/system ]; then
    sudo -n systemctl enable --now tailscaled >/dev/null 2>&1 || true
    echo "TS_DAEMON=systemd"
else
    if [ -c /dev/net/tun ]; then
        TS_TUN_ARG=
    else
        TS_TUN_ARG=--tun=userspace-networking
    fi
    sudo -n mkdir -p /var/lib/tailscale /var/run/tailscale >/dev/null 2>&1 || true
    sudo -n sh -c "nohup tailscaled $TS_TUN_ARG --state=/var/lib/tailscale/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock >/tmp/sparkrun-tailscaled.log 2>&1 </dev/null &" || true
    echo "TS_DAEMON=manual"
    n=0
    while [ $n -lt 15 ]; do
        if [ -S /var/run/tailscale/tailscaled.sock ]; then break; fi
        n=$((n + 1))
        sleep 1
    done
fi

if ! sudo -n tailscale up --authkey="$AUTHKEY" --advertise-tags="$TAGS" $HOSTNAME_ARG $EXTRA_ARGS 2>&1; then
    echo "TS_ERROR=up_failed"
    exit 1
fi

TS_IP=$(tailscale ip -4 2>/dev/null | head -n1)
echo "TS_IP=$TS_IP"
echo "TS_OK=1"
