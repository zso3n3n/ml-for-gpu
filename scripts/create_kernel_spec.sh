#!/usr/bin/env bash
# create_kernel_spec.sh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "ERROR: Missing required arguments." >&2
  echo "Usage: $0 <IMAGE> <KNAME> [FOLDER]" >&2
  exit 1
fi

IMAGE="$1"
KNAME="$2"
FOLDER="${3:-}"

REPO="$(pwd)"
HOME_DIR="$(eval echo ~)"  # Expand tilde to actual home path

DOCKER_BIN="$(command -v docker || true)"
if [[ -z "$DOCKER_BIN" ]]; then
  echo "ERROR: docker not found on PATH." >&2
  exit 1
fi

if ! "$DOCKER_BIN" ps >/dev/null 2>&1; then
  echo "ERROR: cannot run 'docker ps'." >&2
  exit 1
fi

KDIR="${HOME_DIR}/.local/share/jupyter/kernels/${KNAME}"
mkdir -p "$KDIR"

# Use absolute paths - no tilde
cat > "${KDIR}/kernel.json" <<JSON
{
  "argv": [
    "${DOCKER_BIN}", "run", "--rm", "-i",
    "--user", "1000:1000",
    "--gpus", "all",
    "--ipc=host",
    "--shm-size=2g",
    "-v", "${HOME_DIR}:${HOME_DIR}",
    "-v", "${HOME_DIR}/cloudfiles/code/ml-for-gpu/classification:/home/rapids/classification",
    "-v", "${HOME_DIR}/cloudfiles/code/ml-for-gpu/utils:/home/rapids/classification/utils",
    "-w", "/home/rapids/classification",
    "${IMAGE}",
    "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"
  ],
  "display_name": "${KNAME}",
  "language": "python",
  "env": { "PYTHONUNBUFFERED": "1", "HOME": "${HOME_DIR}" }
}
JSON

echo "Installed kernelspec at: ${KDIR}"
echo "Done. In VS Code, pick kernel: '${KNAME}'."