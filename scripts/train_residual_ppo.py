# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Train the residual PPO policy on top of the IK chess skeleton.

This script wires:
  * the registered Isaac gym task (Isaac-Chess-Residual-G1-RightArm-Abs-v0)
    whose env-cfg lives in the patched IsaacLab submodule
    (chess_residual_rl_env_cfg.py — see PR description for the patch),
  * the ChessSkeleton from isaaclab_arena.tasks.g1_chess_residual_rl_task,
  * an rsl_rl PPO runner.

At every env.step the action manager receives
    a_t = a_skeleton(env, phase_state, wrist_xyz) + clip(policy(obs), ±RESIDUAL_CLIP_RAD)

so the policy only learns the residual correction. Both branches address the
same 14-DOF (right arm + hand) action space; the skeleton supplies the
gross IK trajectory, the policy supplies the contact/hold correction.

Usage (Hippocampus, RTX 5090):

    ./submodules/IsaacLab/isaaclab.sh -p scripts/train_residual_ppo.py \\
        --task Isaac-Chess-Residual-G1-RightArm-Abs-v0 \\
        --num_envs 4096 --max_iterations 8000 --headless \\
        --bc_warmstart /home/ray/models/chess-residual-bc-init.pt \\
        --logger wandb
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -- args + AppLauncher MUST come first --
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Chess-Residual-G1-RightArm-Abs-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=8000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bc_warmstart", type=str, default=None,
                    help="Path to a BC-pretrained policy (.pt) to initialize the residual.")
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"])
parser.add_argument("--run_name", type=str, default="residual_ppo")
parser.add_argument("--checkpoint_dir", type=str, default="/home/ray/models/chess-residual-ppo")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- safe to import the rest --
import os
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab_arena.tasks.g1_chess_residual_rl_task import (
    RESIDUAL_CLIP_RAD,
    ChessSkeleton,
    SkeletonState,
    apply_residual,
)
from isaaclab.managers import SceneEntityCfg


def _ppo_runner_cfg(run_name: str, logger: str) -> dict:
    """rsl_rl PPO config — tuned for ~14-DOF residual policy."""
    return {
        "seed": args.seed,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "empirical_normalization": True,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.5,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "experiment_name": run_name,
            "run_name": run_name,
            "logger": logger if logger != "none" else "tensorboard",
            "wandb_project": "chess_residual_rl",
            "neptune_project": "",
            "save_interval": 100,
            "checkpoint": -1,
            "load_run": -1,
            "resume": False,
        },
    }


class ResidualActionWrapper(gym.Wrapper):
    """Wraps the env so the policy sees residual-only action space.

    On every step, we read piece + target + wrist state, query the skeleton
    for its joint-target action, sum with the (clipped) policy residual,
    and pass to the underlying env.
    """

    def __init__(self, env, skeleton: ChessSkeleton, wrist_body_name: str = "right_wrist_yaw_link"):
        super().__init__(env)
        self.skeleton = skeleton
        self.wrist_body_name = wrist_body_name
        self.state = SkeletonState.new(num_envs=env.unwrapped.num_envs)
        # The skeleton produces a Cartesian wrist target; the underlying env's
        # action manager (PINK IK) maps that to joint targets. The policy
        # residual is added in joint space inside the env's action manager.
        # We expose only the residual to the policy.
        self.action_space = gym.spaces.Box(
            low=-RESIDUAL_CLIP_RAD,
            high=RESIDUAL_CLIP_RAD,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.state.reset()
        return obs, info

    def step(self, residual_action):
        underlying = self.env.unwrapped
        # Read wrist xyz to drive phase advancement.
        robot = underlying.scene["robot"]
        bidx = robot.body_names.index(self.wrist_body_name)
        wrist_xyz = robot.data.body_pos_w[:, bidx]
        skel_target_xyz, skel_grasp = self.skeleton.query(underlying, self.state, wrist_xyz)

        # Encode skeleton target into the env's action vector via the env's
        # own action-manager IK (Pink): the env expects (xyz, quat_wxyz, grasp)
        # for the right wrist when using G1_UPPER_BODY_IK_ACTION_CFG. The
        # patched chess_residual_rl_env_cfg.py exposes this as the right_arm_ik
        # action term; the residual is added in-place by the action manager.
        skel_action = self._encode_skeleton_action(skel_target_xyz, skel_grasp)
        action = apply_residual(skel_action, torch.as_tensor(residual_action, device=skel_action.device))
        return self.env.step(action)

    def _encode_skeleton_action(self, xyz: torch.Tensor, grasp: torch.Tensor) -> torch.Tensor:
        """Pack (target xyz, fixed orientation, grasp flag) into the env's action vector.

        The convention matches G1_UPPER_BODY_IK_ACTION_CFG used by the
        placement env. Action layout: [right_wrist_xyz (3), right_wrist_quat_wxyz (4), right_hand_grasp (1), ...].
        Remaining left-arm/left-hand entries are filled with the env's
        current configuration so only the right arm moves.
        """
        underlying = self.env.unwrapped
        num_envs = xyz.shape[0]
        act = torch.zeros(num_envs, self.env.action_space.shape[0], device=xyz.device, dtype=torch.float32)
        # Standard top-down grasp; the patched env cfg validates this layout.
        quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=xyz.device).expand(num_envs, 4)
        act[:, 0:3] = xyz - underlying.scene.env_origins
        act[:, 3:7] = quat
        act[:, 7] = grasp.float()
        return act


def main():
    # Build env cfg + env.
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # Attach skeleton + residual wrapper.
    skeleton = ChessSkeleton(
        piece_cfg=SceneEntityCfg("object"),
        target_cfg=SceneEntityCfg("target"),
    )
    env = ResidualActionWrapper(env, skeleton=skeleton)

    # Build PPO runner.
    runner_cfg = _ppo_runner_cfg(args.run_name, args.logger)
    log_dir = Path(args.checkpoint_dir) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=str(log_dir), device="cuda:0")

    # Optional BC warm-start of the actor.
    if args.bc_warmstart and os.path.isfile(args.bc_warmstart):
        sd = torch.load(args.bc_warmstart, map_location="cuda:0")
        try:
            runner.alg.actor_critic.actor.load_state_dict(sd, strict=False)
            print(f"Loaded BC warm-start from {args.bc_warmstart}")
        except Exception as exc:
            print(f"BC warm-start load failed ({exc}); continuing from scratch")

    runner.learn(num_learning_iterations=args.max_iterations)
    runner.save(str(log_dir / "final.pt"))
    print(f"Training complete; checkpoint at {log_dir / 'final.pt'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
