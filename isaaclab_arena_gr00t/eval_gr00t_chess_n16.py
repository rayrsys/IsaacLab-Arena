"""Closed-loop GR00T N1.6 evaluation for chess piece placement task.

Tests the pre-trained nvidia/GR00T-N1.6-G1-PnPAppleToPlate model on our
chess pick task, or a custom fine-tuned N1.6 checkpoint.

The N1.6 model:
- Uses RELATIVE actions for arms (model outputs absolute after internal conversion)
- Uses ABSOLUTE actions for hands/waist
- Expects full body state (legs, waist, arms, hands)
- Action horizon = 30 (vs 16 in N1.5)
- Single camera: ego_view

Usage:
    cd /home/ray/IsaacLab-Arena/submodules/IsaacLab
    /home/ray/miniconda3/envs/lerobot-arena/bin/python \
        ../../isaaclab_arena_gr00t/eval_gr00t_chess_n16.py \
        --model_path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
        --num_episodes 5 --max_steps 500 --enable_cameras
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to N1.6 model or HF name")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--action_horizon", type=int, default=30, help="Action chunk length (N1.6 default=30)")
parser.add_argument("--denoising_steps", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.manager_based.locomanipulation.pick_place.chess_g1_env_cfg import (
    ChessG1SceneCfg,
    TerminationsCfg,
)


# -- Sim joint indices for each G1 group (from 43dof_joint_space.yaml) --
# These map the 43-DOF articulation order to the GR00T policy groups
SIM_JOINT_GROUPS = {
    "left_leg": [0, 3, 6, 9, 13, 17],   # hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    "right_leg": [1, 4, 7, 10, 14, 18],
    "waist": [2, 5, 8],                   # yaw, roll, pitch
    "left_arm": [11, 15, 19, 21, 23, 25, 27],   # shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
    "right_arm": [12, 16, 20, 22, 24, 26, 28],
    "left_hand": [29, 30, 31, 35, 36, 37, 41],  # index0, middle0, thumb0, index1, middle1, thumb1, thumb2
    "right_hand": [32, 33, 34, 38, 39, 40, 42],
}

# Action groups that map to sim joints (skip base_height_command, navigate_command)
ACTION_JOINT_GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]


@configclass
class EvalActionsCfg:
    """Direct joint position control for upper body (31 DOF)."""
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
            "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
            "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
            "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
            "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )


# 31-DOF action order (matches EvalActionsCfg joint_names above)
ACTION_31DOF_GROUPS = {
    "left_arm": list(range(0, 7)),
    "right_arm": list(range(7, 14)),
    "waist": list(range(14, 17)),
    "left_hand": list(range(17, 20)),   # index0, middle0, thumb0
    "right_hand": list(range(20, 23)),
    "left_hand_1": list(range(23, 26)), # index1, middle1, thumb1
    "right_hand_1": list(range(26, 29)),
    "left_thumb_2": [29],
    "right_thumb_2": [30],
}


@configclass
class EvalObsCfg:
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
class EvalSceneCfg(ChessG1SceneCfg):
    top_cam = None
    def __post_init__(self):
        super().__post_init__()


@configclass
class ChessEvalEnvCfg(ManagerBasedRLEnvCfg):
    scene: EvalSceneCfg = EvalSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    actions: EvalActionsCfg = EvalActionsCfg()
    observations: EvalObsCfg = EvalObsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2


def build_groot_obs(joint_pos_43, camera_rgb, language):
    """Build N1.6 observation dict from sim state."""
    jp = joint_pos_43.cpu().numpy()[0]  # (43,)

    state = {}
    for group, indices in SIM_JOINT_GROUPS.items():
        state[group] = np.array([jp[i] for i in indices], dtype=np.float32).reshape(1, 1, -1)

    rgb_np = camera_rgb.cpu().numpy() if isinstance(camera_rgb, torch.Tensor) else camera_rgb
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)

    obs = {
        "video": {
            "ego_view": rgb_np[0:1].reshape(1, 1, rgb_np.shape[1], rgb_np.shape[2], 3),
        },
        "state": state,
        "language": {
            "annotation.human.task_description": [[language]],
        },
    }
    return obs


def action_dict_to_31dof(action_dict, step_idx):
    """Convert N1.6 action dict to 31-DOF sim action tensor.

    action_dict keys: left_arm(7), right_arm(7), left_hand(7), right_hand(7), waist(3),
                      base_height_command(1), navigate_command(3)

    Our 31-DOF action order: left_arm(7), right_arm(7), waist(3),
                              left_hand(3), right_hand(3), left_hand_1(3), right_hand_1(3),
                              left_thumb_2(1), right_thumb_2(1)
    """
    target = np.zeros(31, dtype=np.float32)

    # Arms (7 DOF each) - direct mapping
    la = action_dict["left_arm"][0, step_idx]   # (7,)
    ra = action_dict["right_arm"][0, step_idx]
    target[0:7] = la
    target[7:14] = ra

    # Waist (3 DOF)
    w = action_dict["waist"][0, step_idx]  # (3,)
    target[14:17] = w

    # Hands: GR00T outputs 7 DOF per hand, our sim has 7 per hand split across indices
    # GR00T hand order: index_0, middle_0, thumb_0, index_1, middle_1, thumb_1, thumb_2
    lh = action_dict["left_hand"][0, step_idx]   # (7,)
    rh = action_dict["right_hand"][0, step_idx]

    # first 3: index_0, middle_0, thumb_0
    target[17:20] = lh[:3]
    target[20:23] = rh[:3]
    # next 3: index_1, middle_1, thumb_1
    target[23:26] = lh[3:6]
    target[26:29] = rh[3:6]
    # last 1: thumb_2
    target[29] = lh[6]
    target[30] = rh[6]

    return target


def main():
    # Create env first
    env_cfg = ChessEvalEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("Env created, loading GR00T N1.6 model...")

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.UNITREE_G1,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"GR00T N1.6 model loaded! (denoising_steps={args.denoising_steps})")

    language = "Pick up the chess piece and place it on the green square."
    action_chunk = None
    action_idx = 0

    import os
    video_dir = "/home/ray/eval_videos_n16"
    os.makedirs(video_dir, exist_ok=True)

    successes = 0
    for ep in range(args.num_episodes):
        env.reset()
        action_chunk = None
        action_idx = 0
        ep_frames = []

        for step in range(args.max_steps):
            robot = env.scene["robot"]
            joint_pos = robot.data.joint_pos.clone()
            cam = env.scene["robot_head_cam"]
            cam_rgb = cam.data.output["rgb"][..., :3]

            if action_chunk is None or action_idx >= args.action_horizon:
                groot_obs = build_groot_obs(joint_pos, cam_rgb, language)
                action_dict, _ = policy.get_action(groot_obs)

                if step == 0 and ep == 0:
                    print(f"  [DEBUG] Action dict keys: {list(action_dict.keys())}")
                    for k, v in action_dict.items():
                        arr = np.array(v)
                        print(f"  [DEBUG]   {k}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

                # Convert to 31-DOF action sequence
                horizon = action_dict["left_arm"].shape[1]
                action_seq = []
                for h in range(horizon):
                    a31 = action_dict_to_31dof(action_dict, h)
                    action_seq.append(a31)
                action_chunk = torch.tensor(np.stack(action_seq), dtype=torch.float32, device=joint_pos.device).unsqueeze(0)
                action_idx = 0

                if step == 0 and ep == 0:
                    print(f"  [DEBUG] action_chunk shape: {action_chunk.shape}")
                    print(f"  [DEBUG] action_chunk[0,0,:7] (left_arm): {action_chunk[0,0,:7].tolist()}")
                    print(f"  [DEBUG] action_chunk[0,0,7:14] (right_arm): {action_chunk[0,0,7:14].tolist()}")

            target_pos = action_chunk[:, action_idx, :]
            action_idx += 1

            target_gpu = target_pos.to(joint_pos.device)
            obs, rew, terminated, truncated, info = env.step(target_gpu)

            if step < 3 and ep == 0:
                jp_after = robot.data.joint_pos[0].cpu()
                jp_before = joint_pos[0].cpu()
                delta = (jp_after - jp_before).abs()
                print(f"  [DEBUG] step={step} joint_delta max={delta.max():.6f} mean={delta.mean():.6f}")

            if step % 4 == 0:
                head_frame = cam_rgb[0].cpu().numpy() if isinstance(cam_rgb, torch.Tensor) else cam_rgb[0]
                if head_frame.dtype != np.uint8:
                    head_frame = (head_frame * 255).clip(0, 255).astype(np.uint8)
                ep_frames.append(head_frame)

            obj = env.scene["object"]
            tgt = env.scene["target"]
            dist = torch.norm(obj.data.root_pos_w[:, :3] - tgt.data.root_pos_w[:, :3], dim=1)
            vel = torch.norm(obj.data.root_vel_w[:, :3], dim=1)
            success = (dist < 0.15) & (vel < 0.5)

            if success.any():
                print(f"  Episode {ep}: SUCCESS at step {step}! dist={dist.item():.3f}")
                successes += 1
                break

            if step % 100 == 0:
                obj_z = obj.data.root_pos_w[0, 2].item()
                print(f"  Episode {ep}, step {step}: obj_dist={dist.item():.3f}, obj_z={obj_z:.3f}")

        if not success.any():
            print(f"  Episode {ep}: TIMEOUT after {args.max_steps} steps, dist={dist.item():.3f}")

        if ep_frames:
            import torchvision
            frames_t = torch.from_numpy(np.stack(ep_frames))
            vid_path = os.path.join(video_dir, f"ep_{ep:02d}.mp4")
            torchvision.io.write_video(vid_path, frames_t, fps=12)
            print(f"  Saved video: {vid_path}")

    print(f"\nResults: {successes}/{args.num_episodes} episodes successful ({100*successes/args.num_episodes:.0f}%)")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
