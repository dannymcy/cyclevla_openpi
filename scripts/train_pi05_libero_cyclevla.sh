#!/usr/bin/env bash
# =============================================================================
# Launch pi05 finetune on the CycleVLA subtask-decomposed LIBERO LeRobot
# dataset (9-D actions: [6D EEF delta, gripper, stop, progress]).
#
# Wraps `uv run scripts/train.py pi05_libero_cyclevla ...` and:
#   * exports three caches off /home onto /hdd2
#   * pins the wandb run dir under openpi/wandb/ (gitignored)
#
# All TrainConfig defaults come from `pi05_libero_cyclevla` registered in
# src/openpi/training/config.py (the entry right after `pi05_libero`).
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   scripts/train_pi05_libero_cyclevla.sh <exp_name> [tyro overrides...]
#
# The first positional arg <exp_name> names the checkpoint subdir:
#   openpi/checkpoints/pi05_libero_cyclevla/<exp_name>/
# All remaining args are forwarded verbatim to train.py as tyro flags.
#
# Examples:
#   # Smoke test (50 steps, save every 25, clobber any prior run):
#   scripts/train_pi05_libero_cyclevla.sh smoke_test \
#       --num-train-steps 50 --save-interval 25 --overwrite
#
#   # Real run (uses TrainConfig num_train_steps=30000):
#   scripts/train_pi05_libero_cyclevla.sh cyclevla_v1
#
#   # Single-GPU memory tweak (default batch_size=256 is sized for 8xH100):
#   scripts/train_pi05_libero_cyclevla.sh cyclevla_v1 \
#       --batch-size 32 --num-workers 4
#
#   # Multi-GPU sharding (FSDP across 4 GPUs):
#   scripts/train_pi05_libero_cyclevla.sh cyclevla_v1 --fsdp-devices 4
#
#   # Resume the same exp_name from its last checkpoint:
#   scripts/train_pi05_libero_cyclevla.sh cyclevla_v1 --resume
#
# -----------------------------------------------------------------------------
# AVAILABLE CLI OVERRIDES (passed AFTER <exp_name>)
# -----------------------------------------------------------------------------
# Booleans use the tyro `--flag` / `--no-flag` convention.
# See `uv run scripts/train.py pi05_libero_cyclevla --help` for the
# exhaustive auto-generated list (this is a curated subset).
#
# --- Top-level TrainConfig fields ---------------------------------------------
#   --project-name STR        wandb project (default "openpi")
#   --seed INT                RNG seed (42)
#   --batch-size INT          global batch size (256 in this config)
#   --num-workers INT         torch dataloader workers (2)
#   --num-train-steps INT     total steps (30000)
#   --log-interval INT        steps between wandb logs (100)
#   --save-interval INT       steps between checkpoint saves (1000)
#   --keep-period INT         keep every Nth checkpoint forever (5000)
#                             pass --no-keep-period to disable
#   --overwrite               clobber existing exp_name dir
#   --resume                  resume from last checkpoint in exp_name dir
#   --no-wandb-enabled        disable wandb entirely
#   --fsdp-devices INT        shard model across N GPUs (1)
#   --checkpoint-base-dir STR checkpoint root (./checkpoints)
#   --assets-base-dir STR     norm-stats root (./assets)
#   --ema-decay FLOAT         EMA decay (0.999); --no-ema-decay disables EMA
#   --pytorch-training-precision {bfloat16,float32}  pytorch path only
#
# --- Nested dataclass overrides (tyro flattens with dotted prefixes) ---------
#   --lr-schedule.peak-lr FLOAT       peak learning rate (5e-5)
#   --lr-schedule.warmup-steps INT    LR warmup (10000)
#   --lr-schedule.decay-steps INT     cosine decay length (1_000_000)
#   --lr-schedule.decay-lr FLOAT      LR floor after decay (5e-5)
#   --optimizer.clip-gradient-norm FLOAT  gradient clipping (1.0)
#   --model.action-horizon INT        flow-matching chunk length (10)
#                                     do NOT raise above ~10 for our short
#                                     subtask sub-episodes
#   --model.action-dim INT            padded action dim (32, default)
#                                     do NOT lower; our 9-D actions get
#                                     zero-padded to 32 by PadStatesAndActions
#   --model.max-token-len INT         language token length (200 for pi05)
#   --data.extra-delta-transform      keep OFF; our actions are already deltas
#   --data.repo-id STR                LeRobot repo id (cyclevla/libero_decomposed_progress)
#   --weight-loader.checkpoint-uri STR  base checkpoint
#                                     (gs://openpi-assets/checkpoints/pi05_libero/params)
#
# -----------------------------------------------------------------------------
# WANDB MODES
# -----------------------------------------------------------------------------
# The launcher pins WANDB_DIR to openpi/wandb/ but leaves WANDB_MODE unset
# (so online wandb is the default). To run offline on a box without internet
# and sync later, set the env var before invoking:
#
#   # Offline run (writes to openpi/wandb/offline-run-* locally):
#   WANDB_MODE=offline scripts/train_pi05_libero_cyclevla.sh cyclevla_v1
#
#   # Or export it once for the shell session:
#   export WANDB_MODE=offline
#   scripts/train_pi05_libero_cyclevla.sh cyclevla_v1
#
#   # Then later, on a machine with internet, sync the offline runs:
#   wandb login   # one-time, if needed
#   wandb sync /hdd2/kai/openvla-oft/openpi/wandb/offline-run-*
#
#   # Or skip wandb entirely (no local files, no sync):
#   scripts/train_pi05_libero_cyclevla.sh cyclevla_v1 --no-wandb-enabled
#   # equivalent to: WANDB_MODE=disabled scripts/.../sh cyclevla_v1
#
# When rsync'ing this repo to another GPU box, copy openpi/data/ (the three
# caches) and openpi/assets/ (norm stats) so nothing has to re-download.
# Checkpoints under openpi/checkpoints/ can be left behind unless you want to
# resume there.
#
# =============================================================================


# How to run
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# chmod +x scripts/train_pi05_libero_cyclevla.sh
# scripts/train_pi05_libero_cyclevla.sh CycleVLA_libero_sub_decomposed_progress_pi05_A10 --project-name cyclevla_openpi --batch-size 32 --fsdp-devices 4 --overwrite

# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
# scripts/train_pi05_libero_cyclevla.sh CycleVLA_libero_sub_decomposed_progress_pi05_A100 --project-name cyclevla_openpi --batch-size 128 --fsdp-devices 8 --overwrite

set -euo pipefail

# --- Caches to /hdd2 (these MUST be exported before python imports) ---
export HF_HOME=/hdd2/kai/openvla-oft/openpi/data/huggingface
export HF_LEROBOT_HOME=/hdd2/kai/openvla-oft/openpi/data/lerobot
export OPENPI_DATA_HOME=/hdd2/kai/openvla-oft/openpi/data/openpi

# For training server
# export HF_HOME=/home/to0space/ygy/openvla-oft/openpi/data/huggingface
# export HF_LEROBOT_HOME=/home/to0space/ygy/openvla-oft/openpi/data/lerobot
# export OPENPI_DATA_HOME=/home/to0space/ygy/openvla-oft/openpi/data/openpi
# export WANDB_MODE=offline

# Required positional arg.
EXP_NAME="${1:?usage: $0 <exp_name> [tyro overrides...]}"
shift

cd "$(dirname "$0")/.."  # cd into openpi/

# Pin wandb run dir under openpi/ regardless of where this is invoked from.
# WANDB_MODE is intentionally NOT set here — the caller chooses online (default),
# offline (export WANDB_MODE=offline), or disabled (--no-wandb-enabled).
export WANDB_DIR="$PWD/wandb"

uv run scripts/train.py pi05_libero_cyclevla \
    --exp-name "$EXP_NAME" \
    "$@"
