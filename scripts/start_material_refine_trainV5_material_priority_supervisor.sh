#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/ssd_work/projects/stable-fast-3d}"
cd "${REPO_ROOT}"

printf '[%s] DEPRECATED: use bash datasetscrip/trainv5_supervisor.sh instead\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2

exec bash datasetscrip/trainv5_supervisor.sh "$@"
