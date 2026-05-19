#!/usr/bin/env bash
# Serve the trained pi05 CycleVLA policy over a websocket so the LIBERO eval
# clients in `openvla-oft/experiments/robot/libero/run_libero_eval_openpi_*.py`
# can query it.
#
# The `pi05_libero_cyclevla` config sets `action_dim=9`, so this server returns
# the full 9-dim action [6D EEF delta, gripper, stop s_t, progress p_t] -- the
# stop/progress dims CycleVLA eval depends on. (Stock `pi05_libero` still
# returns 7 dims; the action_dim default is 7.)
#
# Can be run from anywhere: `scripts/serve_openpi_cyclevla.sh`. Requires the
# checkpoint dir to contain both `params/` (weights) and `assets/<asset_id>/`
# (norm stats).
#
# CKPT_DIR and PORT are env-overridable, and JAX picks the GPU from
# CUDA_VISIBLE_DEVICES -- so to run one server per GPU for parallel evaluation,
# launch several with distinct ports, e.g.:
#   CUDA_VISIBLE_DEVICES=0 PORT=8000 scripts/serve_openpi_cyclevla.sh
#   CUDA_VISIBLE_DEVICES=1 PORT=8001 scripts/serve_openpi_cyclevla.sh
set -euo pipefail

# --- Override via env vars, or edit the defaults for your machine ------------
CKPT_DIR="${CKPT_DIR:-/hdd2/kai/openvla-oft/openpi/checkpoints/pi05_libero_cyclevla/CycleVLA_libero_sub_decomposed_progress_pi05_A100}"
PORT="${PORT:-8000}"
# -----------------------------------------------------------------------------

# This script lives in openpi/scripts/; cd up into openpi/ so the relative
# `scripts/serve_policy.py` path below resolves regardless of caller cwd.
cd "$(dirname "$0")/.."

if [[ ! -d "${CKPT_DIR}/params" || ! -d "${CKPT_DIR}/assets" ]]; then
  echo "ERROR: ${CKPT_DIR} must contain both params/ and assets/ subdirs." >&2
  echo "       Finish transferring the checkpoint (see transfer_server_openpi.sh)." >&2
  exit 1
fi

uv run scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config pi05_libero_cyclevla \
  --policy.dir "${CKPT_DIR}"
