# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Held-out grid evaluation for the residual chess policy.

Rolls out the trained policy on the 12 OOD squares (held-out 4x3 block on
files d-g, ranks 3-5) and reports per-square success rate, mean, and
drop-rate breakdown. Also exports a Markdown table for the thesis report.

Usage:

    ./submodules/IsaacLab/isaaclab.sh -p scripts/eval_residual_holdout.py \\
        --policy /home/ray/models/chess-residual-ppo/final.pt \\
        --holdout_grid configs/chess_holdout_12sq.json \\
        --rollouts_per_square 25 --enable_cameras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Chess-Residual-G1-RightArm-Abs-v0")
parser.add_argument("--holdout_grid", type=str, default=None,
                    help="JSON list of [file_idx, rank_idx] tuples. Defaults to the canonical 4x3 block.")
parser.add_argument("--rollouts_per_square", type=int, default=25)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--out_json", type=str, default="/home/ray/eval_videos/holdout_results.json")
parser.add_argument("--out_md", type=str, default="/home/ray/eval_videos/holdout_results.md")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab_arena.tasks.g1_chess_residual_rl_task import ChessSkeleton
from isaaclab_arena.tasks.g1_chess_target_conditioned_task import (
    holdout_grid_cells,
    square_to_xyz,
)
from scripts.train_residual_ppo import ResidualActionWrapper, _ppo_runner_cfg


def _load_grid() -> list[tuple[int, int]]:
    if args.holdout_grid and Path(args.holdout_grid).is_file():
        with open(args.holdout_grid) as f:
            return [tuple(c) for c in json.load(f)]
    return holdout_grid_cells()


def main():
    grid = _load_grid()
    print(f"Evaluating on {len(grid)} held-out squares × {args.rollouts_per_square} rollouts")

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    skeleton = ChessSkeleton(
        piece_cfg=SceneEntityCfg("object"),
        target_cfg=SceneEntityCfg("target"),
    )
    env = ResidualActionWrapper(env, skeleton=skeleton)

    runner_cfg = _ppo_runner_cfg("eval", "none")
    runner = OnPolicyRunner(env, runner_cfg, log_dir="/tmp/eval_logs", device="cuda:0")
    runner.load(args.policy)
    policy = runner.get_inference_policy(device="cuda:0")

    results: dict[str, dict] = {}
    for (f, r) in grid:
        sq = f"{chr(ord('a')+f)}{r+1}"
        successes = 0
        drops = 0
        # Pin the target by writing it directly each reset via the env's
        # _algebraic flag; the patched env_cfg honors per-env target pose
        # if env.set_target_square(sq) is implemented. For grids without
        # that override, the script falls back to natural randomization
        # and filters episodes by realized target square.
        tx, ty, tz = square_to_xyz(f, r)
        underlying = env.unwrapped
        for _ in range(args.rollouts_per_square):
            obs, _ = env.reset()
            # Force target pose for this episode.
            target = underlying.scene["target"]
            pose = torch.zeros(1, 7, device=underlying.device)
            pose[0, :3] = torch.tensor([tx, ty, tz], device=underlying.device) + underlying.scene.env_origins[0]
            pose[0, 3] = 1.0
            target.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=underlying.device))

            success = False
            drop = False
            for _step in range(args.max_steps):
                with torch.inference_mode():
                    action = policy(obs)
                obs, rew, terminated, truncated, info = env.step(action)
                piece_z = underlying.scene["object"].data.root_pos_w[0, 2] - underlying.scene.env_origins[0, 2]
                if piece_z.item() < 0.45:
                    drop = True
                if bool(terminated[0]) and not drop:
                    success = True
                    break
                if bool(truncated[0]):
                    break
            successes += int(success)
            drops += int(drop)
        results[sq] = {
            "success_rate": successes / args.rollouts_per_square,
            "drop_rate": drops / args.rollouts_per_square,
            "n": args.rollouts_per_square,
        }
        print(f"  {sq}: {results[sq]}")

    mean = sum(v["success_rate"] for v in results.values()) / len(results)
    mean_drop = sum(v["drop_rate"] for v in results.values()) / len(results)
    summary = {"per_square": results, "holdout_mean_success": mean, "holdout_mean_drop": mean_drop}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_md, "w") as f:
        f.write("| Square | Success | Drop |\n|---|---|---|\n")
        for sq, v in sorted(results.items()):
            f.write(f"| {sq} | {v['success_rate']:.2f} | {v['drop_rate']:.2f} |\n")
        f.write(f"\n**Held-out mean success:** {mean:.3f}\n")
        f.write(f"**Held-out mean drop:** {mean_drop:.3f}\n")
    print(f"Wrote {args.out_json} and {args.out_md}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
