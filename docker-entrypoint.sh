#!/bin/sh
set -eu

# Named volumes created by older images can contain root-owned cache metadata.
# Migrate each volume once, then run the application without root privileges.
if [ "$(id -u)" -eq 0 ]; then
    for directory in /cache/huggingface /app/outputs; do
        marker="${directory}/.omnivoice-owner-10001"
        if [ ! -e "${marker}" ]; then
            chown -R 10001:10001 "${directory}"
            touch "${marker}"
            chown 10001:10001 "${marker}"
        fi
    done

    exec setpriv \
        --reuid=10001 \
        --regid=10001 \
        --init-groups \
        -- "$@"
fi

exec "$@"
