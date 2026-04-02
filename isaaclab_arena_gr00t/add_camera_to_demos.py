"""Replay recorded demos and capture camera frames (multi-camera).

Replays joint positions from HDF5 demos through the sim and saves
camera RGB observations back into a new HDF5 file. Supports both
head camera and right wrist camera.

Usage:
    cd /home/ray/IsaacLab-Arena/submodules/IsaacLab
    conda activate lerobot-arena
    ./isaaclab.sh -p /home/ray/IsaacLab-Arena/isaaclab_arena_gr00t/add_camera_to_demos.py \
        --input_file /home/ray/datasets/chess_v3/chess_demos_v3.hdf5 \
        --output_file /home/ray/datasets/chess_v3/chess_demos_v3_multicam.hdf5 \
        --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py
import torch
import numpy as np
from tqdm import tqdm

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


# Camera definitions: (scene_key, hdf5_dataset_name)
CAMERAS = [
    ("robot_head_cam", "robot_head_cam_rgb"),
    ("top_cam", "top_cam_rgb"),
]


@configclass
class ReplayActionsCfg:
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


def capture_camera(env, scene_key):
    """Capture RGB frame from a camera sensor."""
    cam = env.scene[scene_key]
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgb


def main():
    f_in = h5py.File(args.input_file, "r")
    f_out = h5py.File(args.output_file, "w")

    demos = sorted(f_in["data"].keys())
    print(f"Processing {len(demos)} demos from {args.input_file}")
    print(f"Capturing cameras: {[name for _, name in CAMERAS]}")

    env_cfg = ReplayEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    for demo_name in tqdm(demos):
        traj = f_in["data"][demo_name]
        joint_pos_seq = traj["obs"]["robot_joint_pos"][:]
        obj_pos_init = traj["obs"]["object_pos"][0]
        num_steps = len(joint_pos_seq)

        # Copy all existing data (recursively handle groups)
        grp = f_out.create_group(f"data/{demo_name}")

        def copy_group(src, dst, skip_cam=False):
            for key in src.keys():
                item = src[key]
                if isinstance(item, h5py.Group):
                    sub = dst.create_group(key)
                    copy_group(item, sub, skip_cam=(key == "obs"))
                elif isinstance(item, h5py.Dataset):
                    # Skip old camera data in obs — we'll re-capture
                    if skip_cam and "cam_rgb" in key:
                        continue
                    dst.create_dataset(key, data=item[:])

        copy_group(traj, grp)

        # Copy attributes
        for attr_key, attr_val in traj.attrs.items():
            grp.attrs[attr_key] = attr_val

        # Reset env and set object position
        env.reset()
        obj = env.scene["object"]
        obj_pos_t = torch.tensor(obj_pos_init, dtype=torch.float32, device=device).unsqueeze(0)
        obj_pos_w = obj_pos_t + env.scene.env_origins[0:1]
        obj_quat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device)
        obj_pose = torch.cat([obj_pos_w, obj_quat], dim=1)
        obj.write_root_pose_to_sim(obj_pose, env_ids=torch.tensor([0], device=device))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=torch.tensor([0], device=device))

        # Replay and capture all cameras
        all_cam_frames = {hdf5_name: [] for _, hdf5_name in CAMERAS}

        for step in range(num_steps):
            jp = torch.tensor(joint_pos_seq[step], dtype=torch.float32, device=device).unsqueeze(0)
            env.step(jp)

            for scene_key, hdf5_name in CAMERAS:
                rgb = capture_camera(env, scene_key)
                all_cam_frames[hdf5_name].append(rgb)

        # Save all camera streams
        for _, hdf5_name in CAMERAS:
            frames = np.stack(all_cam_frames[hdf5_name])
            grp["obs"].create_dataset(hdf5_name, data=frames, compression="gzip", compression_opts=4)

        print(f"  {demo_name}: {num_steps} steps, "
              + ", ".join(f"{name}={np.stack(all_cam_frames[name]).shape}" for _, name in CAMERAS))

    f_in.close()
    f_out.close()
    env.close()
    simulation_app.close()
    print(f"\nDone! Saved to {args.output_file}")


if __name__ == "__main__":
    main()
