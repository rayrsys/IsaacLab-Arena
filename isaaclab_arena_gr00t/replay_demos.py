"""Replay mimic-generated demos in the chess environment.

Loads recorded joint positions from HDF5 and sets them directly in sim
so you can visually inspect demo quality.

Usage:
    cd /home/ray/IsaacLab-Arena/submodules/IsaacLab
    conda activate lerobot-arena
    python /home/ray/IsaacLab-Arena/isaaclab_arena_gr00t/replay_demos.py \
        --hdf5_path /home/ray/datasets/chess/chess_mimic_1k.hdf5 \
        --demo_ids 0 25 50 75 96
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5_path", type=str, required=True)
parser.add_argument("--demo_ids", type=int, nargs="+", default=[0, 25, 50, 75, 96])
parser.add_argument("--playback_speed", type=float, default=1.0, help="1.0 = realtime, 0.5 = half speed")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py
import time
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

import isaaclab_tasks
import isaaclab_tasks.manager_based.locomanipulation.pick_place
from isaaclab_tasks.manager_based.locomanipulation.pick_place.chess_g1_env_cfg import (
    ChessG1SceneCfg,
    ObservationsCfg,
    TerminationsCfg,
)
import isaaclab.envs.mdp as base_mdp


@configclass
class ReplayActionsCfg:
    """Direct joint position control for all 43 joints."""
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ReplayEnvCfg(ManagerBasedRLEnvCfg):
    scene: ChessG1SceneCfg = ChessG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    actions: ReplayActionsCfg = ReplayActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None
    events = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2


def main():
    f = h5py.File(args.hdf5_path, "r")
    demos = sorted(f["data"].keys())
    print(f"Loaded {len(demos)} demos from {args.hdf5_path}")

    env_cfg = ReplayEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    dt = env_cfg.sim.dt * env_cfg.decimation  # sim step time
    sleep_time = dt / args.playback_speed

    for demo_idx in args.demo_ids:
        demo_name = f"demo_{demo_idx}"
        if demo_name not in f["data"]:
            print(f"  Skipping {demo_name} — not found")
            continue

        traj = f["data"][demo_name]
        joint_pos_seq = traj["obs"]["robot_joint_pos"][:]
        obj_pos = traj["obs"]["object_pos"][0]
        num_steps = len(joint_pos_seq)
        success = traj.attrs.get("success", False)

        print(f"\n=== Replaying {demo_name}: {num_steps} steps, success={success}, obj_start=({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}) ===")

        env.reset()

        # Set initial object position from demo
        obj = env.scene["object"]
        obj_pos_t = torch.tensor(obj_pos, dtype=torch.float32, device=device).unsqueeze(0)
        obj_pos_w = obj_pos_t + env.scene.env_origins[0:1]
        obj_quat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device)
        obj_pose = torch.cat([obj_pos_w, obj_quat], dim=1)
        obj.write_root_pose_to_sim(obj_pose, env_ids=torch.tensor([0], device=device))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=torch.tensor([0], device=device))

        for step in range(num_steps):
            jp = torch.tensor(joint_pos_seq[step], dtype=torch.float32, device=device).unsqueeze(0)
            env.step(jp)
            time.sleep(sleep_time)

            if step % 50 == 0:
                cur_obj_pos = obj.data.root_pos_w[0, :3] - env.scene.env_origins[0]
                print(f"  step {step}/{num_steps}: obj=({cur_obj_pos[0]:.3f}, {cur_obj_pos[1]:.3f}, {cur_obj_pos[2]:.3f})")

        # Final object position
        final_obj = obj.data.root_pos_w[0, :3] - env.scene.env_origins[0]
        print(f"  Final obj pos: ({final_obj[0]:.3f}, {final_obj[1]:.3f}, {final_obj[2]:.3f})")

        # Pause between demos
        print("  Pausing 3s before next demo...")
        time.sleep(3)

    f.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
