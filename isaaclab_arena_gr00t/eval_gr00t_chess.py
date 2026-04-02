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
parser.add_argument("--right_arm_only", action="store_true", help="Use 14 DOF right arm + hand only model")
parser.add_argument("--denoising_steps", type=int, default=4, help="Number of denoising steps (try 16 for complex tasks)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Now safe to import everything else (after AppLauncher) --
import torch
import numpy as np

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# GR00T imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from gr00t.model.policy import Gr00tPolicy
from isaaclab_arena_gr00t.chess_data_config import UnitreeG1ChessDataConfig, UnitreeG1ChessMultiCamDataConfig, UnitreeG1ChessRightArmDataConfig
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
class EvalObsCfg:
    """Observations without top_cam (removed for VRAM)."""
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
    """Scene with only head camera to fit eval + model in 16GB VRAM."""
    top_cam = None  # Remove top-down camera for eval

    def __post_init__(self):
        super().__post_init__()


@configclass
class ChessEvalEnvCfg(ManagerBasedRLEnvCfg):
    """Eval env config: same scene/obs/terminations as chess env but with direct joint control."""
    scene: EvalSceneCfg = EvalSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
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


def build_groot_obs(joint_pos_sim, camera_rgb, policy_joints_config, sim_state_joints_config, num_envs, language, top_cam_rgb=None):
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

    if top_cam_rgb is not None:
        top_np = top_cam_rgb.cpu().numpy() if isinstance(top_cam_rgb, torch.Tensor) else top_cam_rgb
        if top_np.dtype != np.uint8:
            top_np = (top_np * 255).clip(0, 255).astype(np.uint8)
        obs["video.top_view"] = top_np.reshape(num_envs, 1, top_np.shape[1], top_np.shape[2], 3)

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

    # Create env FIRST so Isaac Sim renderer allocates GPU memory before model
    env_cfg = ChessEvalEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("Env created, loading GR00T model...")

    # Load GR00T policy (after env so renderer has VRAM allocated)
    # Use single-cam config for eval to fit in 16GB VRAM
    if args.right_arm_only:
        data_config = UnitreeG1ChessRightArmDataConfig()
        print("Using RIGHT ARM ONLY (14 DOF) data config")
    else:
        data_config = UnitreeG1ChessDataConfig()
    print(f"Loading GR00T model from {args.model_path}...")
    denoising = getattr(args, 'denoising_steps', 4)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=denoising,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Using denoising_steps={denoising}")
    print("GR00T model loaded!")

    language = "Pick up the chess piece."
    num_envs = 1
    device = "cuda"

    action_chunk = None
    action_idx = 0

    # Video recording
    import os
    video_dir = "/home/ray/eval_videos"
    os.makedirs(video_dir, exist_ok=True)

    successes = 0
    for ep in range(args.num_episodes):
        env.reset()
        action_chunk = None
        action_idx = 0
        ep_frames = []

        for step in range(args.max_steps):
            # Get joint positions (articulation order)
            robot = env.scene["robot"]
            joint_pos = robot.data.joint_pos.clone()

            # Get camera image (head cam only — top cam removed for VRAM)
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

                # For right-arm-only mode, fill non-predicted joints with current positions
                if args.right_arm_only:
                    current_state_sim = JointsAbsPosition(joint_pos.cpu(), sim_state_joints_cfg)
                    current_policy = remap_sim_joints_to_policy_joints(current_state_sim, policy_joints_cfg)
                    horizon = list(action_dict.values())[0].shape[1]
                    for group in ["left_arm", "left_hand", "waist"]:
                        if f"action.{group}" not in action_dict and group in current_policy:
                            vals = current_policy[group]  # shape (1, N)
                            action_dict[f"action.{group}"] = np.tile(vals.reshape(1, 1, -1), (1, horizon, 1))

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

            # Save frame for video
            if step % 4 == 0:  # every 4th step for ~12.5 FPS
                head_frame = cam_rgb[0].cpu().numpy() if isinstance(cam_rgb, torch.Tensor) else cam_rgb[0]
                if head_frame.dtype != np.uint8:
                    head_frame = (head_frame * 255).clip(0, 255).astype(np.uint8)
                ep_frames.append(head_frame)

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

        # Save episode video
        if ep_frames:
            import torchvision
            frames_t = torch.from_numpy(np.stack(ep_frames))
            vid_path = os.path.join(video_dir, f"ep_{ep:02d}.mp4")
            torchvision.io.write_video(vid_path, frames_t, fps=12)
            print(f"  Saved video: {vid_path} ({len(ep_frames)} frames)")

    print(f"\nResults: {successes}/{args.num_episodes} episodes successful ({100*successes/args.num_episodes:.0f}%)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
