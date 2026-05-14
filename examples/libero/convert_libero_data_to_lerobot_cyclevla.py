"""
Convert the CycleVLA subtask-decomposed LIBERO RLDS dataset
(`libero_decomposed_progress`, built by
`rlds_dataset_builder/LIBERO_Decomposed_Progress/`) to LeRobot v2.0 format
for pi05 finetuning.

Differences from the upstream `convert_libero_data_to_lerobot.py`:
  - One TFDS dataset, not four. Stage 3 already merges all four LIBERO suites
    into one set of tfrecords, split per-subtask sub-episode.
  - Loads via `tfds.builder_from_directory(...)` so the
    `LIBERO_Decomposed_Progress` builder class does NOT need to be importable
    in the openpi env (reads `dataset_info.json` / `features.json` on disk).
  - `actions` is 9-D so pi05's flow-matching action expert predicts the
    CycleVLA supervision channels (s_t, p_t) jointly with the 7-D action.

==== Design decisions ====

[1] state stays 8-D, actions becomes 9-D — no coupling between them.
    `state` is the proprioceptive observation LIBERO records from the env
    ([6D EEF pose, 2D gripper state]) and is fixed by data collection.
    `actions` is what the policy predicts; CycleVLA extends the original
    7-D LIBERO action with two supervision dims:
        [Δx, Δy, Δz, Δu, Δv, Δw, gripper, s_t, p_t]
         <----- 6-D EEF delta -----> <-7--> <stop> <progress>
    Dims 0-5 are end-effector pose DELTAS (same convention OFT trains on:
    Stage-3 builder header, `LIBERO_Decomposed_Progress_dataset_builder.py:9-14`;
    OFT transform `prismatic/.../oxe/transforms.py:828-854`).

[2] Image resolution: store at 256x256, do NOT pre-resize to 224.
    Both pipelines down-resize at the transform stage:
      * openpi: `_transforms.ResizeImages(224, 224)` in
        `openpi/src/openpi/training/config.py:119,131,151`.
      * openpi LIBERO policy inputs: 224x224 in
        `openpi/src/openpi/policies/libero_policy.py: make_libero_example`.
      * OFT: 256-stored RLDS resized to 224 in the OXE loader.
    openpi's reported LIBERO numbers (`examples/libero/README.md`: Spatial
    98.8 / Object 98.2 / Goal 98.0 / Ten 92.4) were trained on a 256-stored
    LeRobot dataset resized to 224 in the data pipeline. We follow this.

[3] Gripper is kept RAW — no `invert_gripper_actions`.
    OFT inverts gripper in the dataloader (`1 - clip(g, 0, 1)`,
    `prismatic/.../data_utils.py:128-129`, applied at `transforms.py:831`)
    and de-inverts at eval time. We follow openpi's stock `pi05_libero`
    training trajectory instead: keep the raw RLDS gripper, train pi05 as
    openpi does, and handle any convention swap in a custom LIBERO
    inference script that will live under `openvla-oft/` (separate task).
    Norm stats will recompute per-dim automatically for the 9-D actions.

[4] Subtask language flows into `task`.
    The Stage-3 builder writes the per-subtask string into
    `language_instruction` (not the high-level task description), so each
    LeRobot frame gets the subtask label as its `task`. Each RLDS episode
    is one subtask sub-episode, so `save_episode()` per RLDS-episode
    preserves Stage-3's per-subtask splitting and tail-oversampling.

Usage (run from `openpi/`):

  # Keep the output OFF /home — write to openpi/data/, which is in .gitignore.
  HF_LEROBOT_HOME=/hdd2/kai/openvla-oft/openpi/data/lerobot \\
  uv run examples/libero/convert_libero_data_to_lerobot_cyclevla.py \\
      --data_dir /hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress

Output location:
  Resolved as `$HF_LEROBOT_HOME / cyclevla/libero_decomposed_progress`.

  With the command above:
    /hdd2/kai/openvla-oft/openpi/data/lerobot/cyclevla/libero_decomposed_progress/

  If `HF_LEROBOT_HOME` is unset, lerobot defaults to
  `~/.cache/huggingface/lerobot/` — that lands on /home, which is why we
  override the env var here.
"""

# export HF_LEROBOT_HOME=/hdd2/kai/openvla-oft/openpi/data/lerobot
# uv run examples/libero/convert_libero_data_to_lerobot_cyclevla.py --data_dir /hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress

import glob
import os
import shutil

# Pin the LeRobot dataset root to openpi/data/ (gitignored, on /hdd2) so a
# plain `uv run` does NOT spill into ~/.cache/huggingface/lerobot/ on /home.
# Must be set BEFORE importing lerobot, because the package resolves
# `HF_LEROBOT_HOME` at import time. Using setdefault so a user-exported
# env var still wins.
# os.environ.setdefault(
#     "HF_LEROBOT_HOME", "/hdd2/kai/openvla-oft/openpi/data/lerobot"
# )

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "cyclevla/libero_decomposed_progress"
DATASET_NAME = "libero_decomposed_progress"


def _resolve_builder_dir(data_dir: str) -> str:
    # tfds writes to <data_dir>/<dataset_name>/<version>/. Pick the highest
    # version subdirectory present.
    pattern = os.path.join(data_dir, DATASET_NAME, "*")
    candidates = [p for p in sorted(glob.glob(pattern)) if os.path.isdir(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No version subdirectory under {os.path.join(data_dir, DATASET_NAME)}. "
            f"Run `tfds build` for LIBERO_Decomposed_Progress first."
        )
    return candidates[-1]


def main(data_dir: str, *, push_to_hub: bool = False):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    builder_dir = _resolve_builder_dir(data_dir)
    print(f"[cyclevla] reading TFDS dataset from: {builder_dir}")
    builder = tfds.builder_from_directory(builder_dir)
    raw_dataset = builder.as_dataset(split="train")

    n_episodes = 0
    for episode in raw_dataset:
        for step in episode["steps"].as_numpy_iterator():
            # Build 9-D action: [6D EEF delta, gripper_raw, s_t, p_t].
            # See decision [1] / [3] above.
            action_6d   = step["action"][:6].astype(np.float32)
            gripper_raw = np.float32(step["action"][6])
            s_t         = np.float32(step["is_last"])
            p_t         = np.float32(step["is_terminal"])
            actions_9d  = np.concatenate(
                [action_6d, np.array([gripper_raw, s_t, p_t], dtype=np.float32)]
            )

            dataset.add_frame(
                {
                    "image":       step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state":       step["observation"]["state"].astype(np.float32),
                    "actions":     actions_9d,
                    "task":        step["language_instruction"].decode(),
                }
            )
        dataset.save_episode()
        n_episodes += 1
        if n_episodes % 100 == 0:
            print(f"[cyclevla] wrote {n_episodes} sub-episodes")

    print(f"[cyclevla] done: {n_episodes} sub-episodes -> {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "cyclevla", "subtask", "progress"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
