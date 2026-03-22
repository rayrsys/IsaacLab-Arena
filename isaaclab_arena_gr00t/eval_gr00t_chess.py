"""Closed-loop GR00T evaluation for chess piece placement task.

Uses the built-in Isaac Lab chess env (Isaac-Chess-FixedBase-G1-Abs-v0) with
direct joint position control (no PINK IK), matching how demos were collected.

Usage:
    cd /home/ray/IsaacLab-Arena/submodules/IsaacLab
    ./isaaclab.sh -p ../../isaaclab_arena_gr00t/eval_gr00t_chess.py \
        --model_path /home/ray/models/chess-piece-placement-merged \
        --num_episodes 5 --max_steps 500 --enable_cameras
"""

import argparse
import sys
from pathlib import Path

# -- AppLauncher must be created FIRST before any isaaclab imports --
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to merged GR00T checkpoint")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--action_horizon", type=int, default=16, help="Action chunk length")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Now safe to import everything else (after AppLauncher) --
import torch
import numpy as np

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

# GR00T imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from gr00t.model.policy import Gr00tPolicy
from isaaclab_arena_gr00t.chess_data_config import UnitreeG1ChessDataConfig
from isaaclab_arena_gr00t.data_utils.io_utils import load_robot_joints_config_from_yaml
from isaaclab_arena_gr00t.data_utils.joints_conversion import (
    remap_policy_joints_to_sim_joints,
    remap_sim_joints_to_policy_joints,
)
from isaaclab_arena_gr00t.data_utils.robot_joints import JointsAbsPosition

# Import the chess env scene/obs/termination configs
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.manager_based.locomanipulation.pick_place.chess_g1_env_cfg import (
    ChessG1SceneCfg,
    ObservationsCfg,
    TerminationsCfg,
)


# -- Custom env config that uses direct joint position control (no PINK IK) --

@configclass
class EvalActionsCfg:
    """Direct joint position control for upper body only (31 DOF) — bypasses PINK IK entirely."""
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


@configclass
class ChessEvalEnvCfg(ManagerBasedRLEnvCfg):
    """Eval env config: same scene/obs/terminations as chess env but with direct joint control."""
    scene: ChessG1SceneCfg = ChessG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    actions: EvalActionsCfg = EvalActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2


def build_groot_obs(joint_pos_sim, camera_rgb, policy_joints_config, sim_state_joints_config, num_envs, language):
    """Convert env observations to GR00T input format."""
    joint_pos_state_sim = JointsAbsPosition(joint_pos_sim.cpu(), sim_state_joints_config)
    joint_pos_policy = remap_sim_joints_to_policy_joints(joint_pos_state_sim, policy_joints_config)

    rgb_np = camera_rgb.cpu().numpy() if isinstance(camera_rgb, torch.Tensor) else camera_rgb
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)

    obs = {
        "annotation.human.task_description": [language] * num_envs,
        "video.ego_view": rgb_np.reshape(num_envs, 1, rgb_np.shape[1], rgb_np.shape[2], 3),
        "state.left_arm": joint_pos_policy["left_arm"].reshape(num_envs, 1, -1),
        "state.right_arm": joint_pos_policy["right_arm"].reshape(num_envs, 1, -1),
        "state.left_hand": joint_pos_policy["left_hand"].reshape(num_envs, 1, -1),
        "state.right_hand": joint_pos_policy["right_hand"].reshape(num_envs, 1, -1),
    }
    return obs


def main():
    # Load joint configs
    arena_root = Path(__file__).parent.parent
    policy_joints_cfg = load_robot_joints_config_from_yaml(
        arena_root / "isaaclab_arena_gr00t/config/g1/gr00t_43dof_joint_space.yaml"
    )
    sim_state_joints_cfg = load_robot_joints_config_from_yaml(
        arena_root / "isaaclab_arena_gr00t/config/g1/43dof_joint_space.yaml"
    )
    sim_action_joints_cfg = load_robot_joints_config_from_yaml(
        arena_root / "isaaclab_arena_gr00t/config/g1/31dof_upper_body_joint_space.yaml"
    )

    # Load GR00T policy
    data_config = UnitreeG1ChessDataConfig()
    print(f"Loading GR00T model from {args.model_path}...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("GR00T model loaded!")

    # Create env with direct joint position control (no PINK IK)
    env_cfg = ChessEvalEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    language = "Pick up the chess piece and place it on the green target square."
    num_envs = 1
    device = "cuda"

    action_chunk = None
    action_idx = 0

    successes = 0
    for ep in range(args.num_episodes):
        env.reset()
        action_chunk = None
        action_idx = 0

        for step in range(args.max_steps):
            # Get joint positions (articulation order)
            robot = env.scene["robot"]
            joint_pos = robot.data.joint_pos.clone()

            # Get camera image
            cam = env.scene["robot_head_cam"]
            cam_rgb = cam.data.output["rgb"][..., :3]

            # Check if we need a new action chunk
            if action_chunk is None or action_idx >= args.action_horizon:
                groot_obs = build_groot_obs(
                    joint_pos, cam_rgb, policy_joints_cfg, sim_state_joints_cfg,
                    num_envs, language,
                )
                action_dict = policy.get_action(groot_obs)

                # Debug: print GR00T raw output keys and shapes
                if step == 0 and ep == 0:
                    print(f"  [DEBUG] GR00T action_dict keys: {list(action_dict.keys())}")
                    for k, v in action_dict.items():
                        arr = np.array(v) if not isinstance(v, np.ndarray) else v
                        print(f"  [DEBUG]   {k}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")
                    print(f"  [DEBUG] Current joint_pos shape: {joint_pos.shape}")
                    print(f"  [DEBUG] Current joint_pos[0, :5]: {joint_pos[0, :5].tolist()}")

                action_sim = remap_policy_joints_to_sim_joints(
                    action_dict, policy_joints_cfg, sim_action_joints_cfg, device
                )
                action_chunk = action_sim.get_joints_pos()

                if step == 0 and ep == 0:
                    print(f"  [DEBUG] action_chunk shape: {action_chunk.shape}")
                    print(f"  [DEBUG] action_chunk[0, 0, :5]: {action_chunk[0, 0, :5].tolist()}")
                    min_dim = min(action_chunk.shape[2], joint_pos.shape[1])
                    diff = (action_chunk[0, 0, :min_dim].cpu() - joint_pos[0, :min_dim].cpu()).abs()
                    print(f"  [DEBUG] |target - current| max={diff.max():.4f}, mean={diff.mean():.4f}")

                action_idx = 0

            # Get current action from chunk
            target_pos = action_chunk[:, action_idx, :]
            action_idx += 1

            # Use env.step() so the action manager properly applies joint targets
            target_gpu = target_pos.to(joint_pos.device)
            obs, rew, terminated, truncated, info = env.step(target_gpu)

            # Debug: check if joints changed
            if step < 3 and ep == 0:
                jp_after = robot.data.joint_pos[0].cpu()
                jp_before = joint_pos[0].cpu()
                tp = target_gpu[0].cpu()
                delta = (jp_after - jp_before).abs()
                print(f"  [DEBUG] step={step} joint_delta max={delta.max():.6f} mean={delta.mean():.6f}")
                print(f"  [DEBUG] step={step} larm target[:7]: {tp[:7].tolist()}")
                print(f"  [DEBUG] step={step} rarm target[7:14]: {tp[7:14].tolist()}")

            # Check success
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
                print(f"  Episode {ep}, step {step}: obj_dist={dist.item():.3f}, vel={vel.item():.3f}")

        if not success.any():
            print(f"  Episode {ep}: TIMEOUT after {args.max_steps} steps, dist={dist.item():.3f}")

    print(f"\nResults: {successes}/{args.num_episodes} episodes successful ({100*successes/args.num_episodes:.0f}%)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
