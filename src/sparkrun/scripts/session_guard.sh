#!/bin/bash
# sparkrun session guard — terminate this host's work if the controlling SSH
# session goes away, so a killed `sparkrun` on the control node cannot orphan a
# multi-GB download / image pull / rsync fan-out here.
#
# Remote payloads run via `ssh <host> bash -s`, i.e. WITHOUT a PTY.  On
# disconnect sshd's session process exits but sends no signal to its child (the
# SIGHUP-on-disconnect path is PTY-only), so the payload is merely reparented
# and keeps running, invisible to the control node.  This wrapper detects that
# reparenting and kills the payload's whole process group.
#
# NOTE: consumed via Python — the payload sentinel line below is replaced by the
# payload script (see ``orchestration.ssh.wrap_with_session_guard``).  The
# sentinel is matched as a whole line, and must appear exactly once, so do not
# repeat that token anywhere else in this file (including in a comment).
#
# Like model_distribute.sh this file must contain NO literal curly-brace
# characters (shell functions are therefore avoided in favor of parenthesized
# subshells), so that it stays safe to run through Python str.format().

# Parent at start: the per-session sshd.  Empty when `ps` is unavailable, in
# which case the guard stays inert rather than guessing (see below).
__sr_ppid0="$(ps -o ppid= -p $$ 2>/dev/null | tr -d '[:space:]')"

# `set -m` gives the payload its own process group, so the group kill below
# reaches grandchildren (e.g. uvx -> hf download).  It is turned back OFF as the
# first statement *inside* the payload subshell: with job control still on, a
# payload that backgrounds its own jobs (model_distribute.sh, image_distribute.sh)
# would put each one in a separate process group where the group kill can't
# reach them.
set -m
(
set +m
__SPARKRUN_PAYLOAD__
) &
__sr_job=$!
set +m

# A signal delivered to this shell (e.g. sshd relaying one) tears the payload
# down the same way.
trap 'kill -TERM -$__sr_job 2>/dev/null; sleep 5; kill -KILL -$__sr_job 2>/dev/null; exit 143' HUP INT TERM

__sr_watch=""
if [ -n "$__sr_ppid0" ]; then
    # Watchdog: poll our own parent.  When the ssh client dies, sshd exits and
    # this shell is reparented (to init/systemd), which is the disconnect
    # signal.  The watchdog deliberately prints NOTHING on that path — stderr
    # is the dead channel by then, and a SIGPIPE there would kill the watchdog
    # before it kills the payload.
    (
        while kill -0 "$__sr_job" 2>/dev/null; do
            sleep 2
            __sr_now="$(ps -o ppid= -p $$ 2>/dev/null | tr -d '[:space:]')"
            if [ -n "$__sr_now" ] && [ "$__sr_now" != "$__sr_ppid0" ]; then
                kill -TERM -$__sr_job 2>/dev/null
                sleep 5
                kill -KILL -$__sr_job 2>/dev/null
                exit 0
            fi
        done
    ) </dev/null >/dev/null 2>&1 &
    __sr_watch=$!
fi

wait "$__sr_job"
__sr_rc=$?
if [ -n "$__sr_watch" ]; then
    kill "$__sr_watch" 2>/dev/null
fi
exit "$__sr_rc"
