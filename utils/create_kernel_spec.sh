#!/usr/bin/env bash
# create_kernel_spec.sh
# Usage:
#   ./create_kernel_spec.sh [IMAGE=rapids-cuopt:cuda122] [KNAME=cu-sk-docker]
# Example:
#   ./create_kernel_spec.sh rapids-cuopt:cuda122 cu-sk-docker

set -euo pipefail

IMAGE="${1:-rapids-cuopt:cuda122}"
KNAME="${2:-cu-sk-docker}"
REPO="$(pwd)"   # Hardcoded to current working directory

DOCKER_BIN="$(command -v docker || true)"
if [[ -z "$DOCKER_BIN" ]]; then
  echo "ERROR: docker not found on PATH." >&2
  exit 1
fi

# Must be able to run docker without sudo
if ! "$DOCKER_BIN" ps >/dev/null 2>&1; then
  echo "ERROR: cannot run 'docker ps'. Ensure your user is in the 'docker' group, then re-login:" >&2
  echo "  sudo usermod -aG docker $USER && newgrp docker" >&2
  exit 1
fi

# Validate repo path
if [[ ! -d "$REPO" ]]; then
  echo "ERROR: Current working directory does not exist or is not a directory: $REPO" >&2
  exit 1
fi

KDIR="${HOME}/.local/share/jupyter/kernels/${KNAME}"
mkdir -p "$KDIR"

# Write kernel.json (absolute paths only; no env expansion)
cat > "${KDIR}/kernel.json" <<JSON
{
  "argv": [
    "${DOCKER_BIN}", "run", "--rm", "-i",
    "--gpus", "all",
    "--ipc=host",
    "--shm-size=2g",
    "-v", "{connection_file}:{connection_file}",
    "-v", "${REPO}:/home/rapids/notebooks/ml-for-gpu",
    "-w", "/home/rapids/notebooks/ml-for-gpu",
    "${IMAGE}",
    "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"
  ],
  "display_name": "cu-sk (Docker, RAPIDS+cuOpt)",
  "language": "python",
  "env": { "PYTHONUNBUFFERED": "1" }
}
JSON

# Show result
echo "Installed kernelspec at: ${KDIR}"
jupyter kernelspec list | sed -n "1p; /${KNAME}/p"
echo "Done. In VS Code, pick kernel: '${KNAME}'."
