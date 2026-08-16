#!/bin/bash
set -uo pipefail

# Ensure a Docker image is present on this host.
# Placeholders filled by Python: {image}, {force_pull}
#
# NOTE: this file is consumed via Python str.format(); it must contain NO
# literal curly-brace characters.
#
# FORCE_PULL="1" is `sparkrun run --rebuild` for a registry image: skip the
# presence check and re-pull unconditionally.  The presence check is
# metadata-only -- `docker image inspect` is satisfied by an image whose
# content blobs are missing -- so it is exactly what a user reaches for
# --rebuild to bypass when the local copy is stale or incomplete.
FORCE_PULL="{force_pull}"

if [ "$FORCE_PULL" != "1" ] && docker image inspect "{image}" >/dev/null 2>&1; then
    echo "Image already available: {image}"
    exit 0
fi

if [ "$FORCE_PULL" = "1" ]; then
    echo "Force pull requested for image: {image}..."
else
    echo "Pulling image: {image}..."
fi
docker pull "{image}"
