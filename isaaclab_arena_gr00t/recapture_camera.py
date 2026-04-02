"""Replay recorded demos and capture camera images into HDF5.

Reads joint positions from HDF5, replays them in sim with the camera enabled,
and writes robot_head_cam_rgb into the HDF5 (creates if missing, overwrites if present).

Supports both old format (obs/object_pos + obs/object_rot) and new format (obs/object, 13D).

Usage:
    cd /home/ray/IsaacLab-Arena/submodules/IsaacLab
    ./isaaclab.sh -p ../../isaaclab_arena_gr00t/recapture_camera.py \
        --input_file /home/ray/datasets/chess/chess_right_arm_v1.hdf5 \
        --enable_cameras --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, default=None, help="Output HDF5 path (default: overwrite input)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py
import numpy as np
import shutil
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.manager_based.locomanipulation.pick_place.chess_g1_env_cfg import (
    ChessG1SceneCfg,
    EventsCfg,
)


@configclass
class RecaptureObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_head_cam_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot_head_cam"), "data_type": "rgb", "normalize": False},
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()


@configclass
class RecaptureActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False,
    )


@configclass
class RecaptureEnvCfg(ManagerBasedRLEnvCfg):
    scene: ChessG1SceneCfg = ChessG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    actions: RecaptureActionsCfg = RecaptureActionsCfg()
    observations: RecaptureObsCfg = RecaptureObsCfg()
    terminations = None
    events: EventsCfg = EventsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2


env = ManagerBasedRLEnv(cfg=RecaptureEnvCfg())

# If output_file specified, copy input first then modify the copy
output_path = args.output_file or args.input_file
if args.output_file and args.output_file != args.input_file:
    shutil.copy2(args.input_file, args.output_file)
    print(f"Copied {args.input_file} -> {args.output_file}")

f = h5py.File(output_path, "r+")
demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
print(f"Found {len(demo_keys)} demos in {output_path}")

for demo_idx, demo_key in enumerate(demo_keys):
    demo = f[f"data/{demo_key}"]
    joint_pos_seq = demo["obs"]["robot_joint_pos"][:]
    n_steps = len(joint_pos_seq)

    # Parse object pose — support both old (object_pos/object_rot) and new (object, 13D) formats
    if "object_pos" in demo["obs"]:
        obj_pos_seq = demo["obs"]["object_pos"][:]
        obj_rot_seq = demo["obs"]["object_rot"][:]
    elif "object" in demo["obs"]:
        obj_data = demo["obs"]["object"][:]
        obj_pos_seq = obj_data[:, :3]
        obj_rot_seq = obj_data[:, 3:7]
    else:
        raise KeyError(f"No object position data found in {demo_key}/obs")

    print(f"[{demo_idx+1}/{len(demo_keys)}] Replaying {demo_key}: {n_steps} steps...")

    env.reset()

    # Set object to its original position from the demo
    obj = env.scene["object"]
    init_pos = torch.tensor(obj_pos_seq[0], dtype=torch.float32, device=env.device).unsqueeze(0)
    init_rot = torch.tensor(obj_rot_seq[0], dtype=torch.float32, device=env.device).unsqueeze(0)
    init_pos_w = init_pos + env.scene.env_origins[:1]
    obj.write_root_pose_to_sim(torch.cat([init_pos_w, init_rot], dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    print(f"  Object placed at {obj_pos_seq[0]}")

    new_frames = []

    for step in range(n_steps):
        jp = torch.tensor(joint_pos_seq[step], dtype=torch.float32, device=env.device).unsqueeze(0)
        obs, _, _, _, _ = env.step(jp)

        cam = env.scene["robot_head_cam"]
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        new_frames.append(rgb)

    new_frames = np.stack(new_frames, axis=0)

    # Create or overwrite camera data in HDF5
    cam_key = f"data/{demo_key}/obs/robot_head_cam_rgb"
    if cam_key in f:
        old_shape = f[cam_key].shape
        if new_frames.shape[0] < old_shape[0]:
            pad = np.tile(new_frames[-1:], (old_shape[0] - new_frames.shape[0], 1, 1, 1))
            new_frames = np.concatenate([new_frames, pad], axis=0)
        f[cam_key][...] = new_frames[:old_shape[0]]
        print(f"  Overwrote {demo_key}/obs/robot_head_cam_rgb: {new_frames.shape}")
    else:
        demo["obs"].create_dataset("robot_head_cam_rgb", data=new_frames, chunks=True, compression="gzip")
        print(f"  Created {demo_key}/obs/robot_head_cam_rgb: {new_frames.shape}")

f.close()
print("Done! All camera data captured.")

env.close()
simulation_app.close()
