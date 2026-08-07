#!/bin/bash
# Probe whether an image's ENTRYPOINT passes sparkrun's command through.
#
# sparkrun always appends its launcher as CMD *arguments*
# (`docker run <image> bash -c <b64 cmd>`), never as the entrypoint.  An image
# whose ENTRYPOINT consumes its args -- e.g. ENTRYPOINT ["vllm","serve"] --
# therefore eats the launcher and runs something else entirely, while a
# passthrough wrapper that ends in `exec "$@"` -- e.g.
# /opt/nvidia/nvidia_entrypoint.sh, which nearly every NGC-derived image ships
# -- is completely harmless.  The two are indistinguishable by inspection, so
# this runs the *real* argv shape and looks for a sentinel.
set -uo pipefail

ENTRYPOINT=$(docker image inspect --format '{{{{json .Config.Entrypoint}}}}' {image} 2>/dev/null || true)
echo "SPARKRUN_ENTRYPOINT=$ENTRYPOINT"

case "$ENTRYPOINT" in
    ''|null|'[]')
        # No ENTRYPOINT at all: the appended CMD *is* the argv, so there is
        # nothing that could consume it.  Skip the container start entirely --
        # this is the cheap exit for the majority of images.
        echo "SPARKRUN_PROBE=absent"
        exit 0
        ;;
esac

# Two deliberate choices guard against a false "pass":
#   * stdout only (2>/dev/null) -- a consuming entrypoint frequently echoes the
#     argv it rejected back on stderr, which would match a literal sentinel.
#   * the sentinel is *computed*, not literal, so even an echoed-back argv
#     cannot produce it -- only a shell that actually ran the command can.
OUTPUT=$(docker run --rm {accel_opts} {image} bash -c '{probe_command}' 2>/dev/null || true)
case "$OUTPUT" in
    *{probe_token}*)
        echo "SPARKRUN_PROBE=pass"
        exit 0
        ;;
esac

# The command did not run -- but "the entrypoint ate it" is only one of the
# reasons why.  A stale CDI spec, an unavailable GPU, a missing bash, or any
# other broken-container-start looks identical from here, and reporting those
# as an entrypoint problem would block launches over an unrelated fault.
# Re-run byte-identically except with the ENTRYPOINT cleared: only if *that*
# succeeds is the entrypoint provably the cause -- which also makes the fix
# sparkrun recommends a verified one rather than an inferred one.
CLEARED=$(docker run --rm --entrypoint '' {accel_opts} {image} bash -c '{probe_command}' 2>/dev/null || true)
case "$CLEARED" in
    *{probe_token}*) echo "SPARKRUN_PROBE=fail" ;;
    *) echo "SPARKRUN_PROBE=unknown" ;;
esac
exit 0
