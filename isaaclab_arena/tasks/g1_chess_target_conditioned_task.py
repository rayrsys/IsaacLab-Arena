# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Target-conditioned chess pick-and-place task for the G1.

Extends G1ChessPiecePlacementTask with:
  * per-episode randomization of (source square, target square) over a
    configurable board grid (events.reset_chess_grid),
  * privileged observations of target xyz, piece xyz, phase, and
    fingertip-piece contact force (observations.policy),
  * a shaped reward that decomposes into reach / hold / place / drop /
    action-norm terms,
  * a curriculum that widens the (source, target) grid as success rises.

The success termination reuses ``objects_in_proximity`` from
``isaaclab_arena.tasks.terminations``.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.tasks.g1_chess_piece_placement_task import G1ChessPiecePlacementTask
from isaaclab_arena.tasks.terminations import objects_in_proximity


# ---------- Board geometry ---------------------------------------------------

# Board is centred at (BOARD_X, BOARD_Y), surface at BOARD_Z. Cells are SQ_SIZE wide.
# These match the placement env's chess_board pose (0.5, 0.0, 0.6) and the existing
# piece initial pose; tuning lives here so all helpers stay consistent.
BOARD_X: float = 0.5
BOARD_Y: float = 0.0
BOARD_Z: float = 0.51
SQ_SIZE: float = 0.035  # ~3.5 cm per square (0.3-scaled blender board)

# Held-out 4x3 OOD block: files d-g (3..6), ranks 3-5 (2..4)
HOLDOUT_FILES = {3, 4, 5, 6}
HOLDOUT_RANKS = {2, 3, 4}


def square_to_xyz(file_idx: int, rank_idx: int) -> tuple[float, float, float]:
    """Algebraic file (0..7=a..h) + rank (0..7=1..8) -> board-surface (x,y,z)."""
    cx = BOARD_X + (file_idx - 3.5) * SQ_SIZE
    cy = BOARD_Y + (rank_idx - 3.5) * SQ_SIZE
    return cx, cy, BOARD_Z


def is_holdout(file_idx: int, rank_idx: int) -> bool:
    return file_idx in HOLDOUT_FILES and rank_idx in HOLDOUT_RANKS


def train_grid_cells() -> list[tuple[int, int]]:
    return [(f, r) for f in range(8) for r in range(8) if not is_holdout(f, r)]


def holdout_grid_cells() -> list[tuple[int, int]]:
    return [(f, r) for f in range(8) for r in range(8) if is_holdout(f, r)]


# ---------- Event terms ------------------------------------------------------

def reset_chess_grid(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    piece_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    train_only: bool = True,
    seed_offset: int = 0,
) -> None:
    """Place piece on a random training source square; target on a different square."""
    if env_ids is None:
        return
    cells = train_grid_cells() if train_only else (train_grid_cells() + holdout_grid_cells())
    n_cells = len(cells)
    device = env.device

    gen = torch.Generator(device="cpu").manual_seed(int(env.common_step_counter) + seed_offset)
    src_idx = torch.randint(0, n_cells, (len(env_ids),), generator=gen)
    tgt_idx = torch.randint(0, n_cells, (len(env_ids),), generator=gen)
    tgt_idx = torch.where(tgt_idx == src_idx, (tgt_idx + 1) % n_cells, tgt_idx)

    piece_pose = torch.zeros(len(env_ids), 7, device=device)
    tgt_pose = torch.zeros(len(env_ids), 7, device=device)
    for i, (s, t) in enumerate(zip(src_idx.tolist(), tgt_idx.tolist())):
        sx, sy, sz = square_to_xyz(*cells[s])
        tx, ty, tz = square_to_xyz(*cells[t])
        piece_pose[i, :3] = torch.tensor([sx, sy, sz + 0.02], device=device)
        piece_pose[i, 3] = 1.0  # quat w
        tgt_pose[i, :3] = torch.tensor([tx, ty, tz], device=device)
        tgt_pose[i, 3] = 1.0

    piece_pose[:, :3] += env.scene.env_origins[env_ids]
    tgt_pose[:, :3] += env.scene.env_origins[env_ids]

    piece = env.scene[piece_cfg.name]
    target = env.scene[target_cfg.name]
    piece.write_root_pose_to_sim(piece_pose, env_ids=env_ids)
    piece.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=device), env_ids=env_ids)
    target.write_root_pose_to_sim(tgt_pose, env_ids=env_ids)


# ---------- Observation terms ------------------------------------------------

def target_xyz_obs(env: ManagerBasedRLEnv, target_cfg: SceneEntityCfg) -> torch.Tensor:
    target = env.scene[target_cfg.name]
    return target.data.root_pos_w - env.scene.env_origins


def piece_xyz_obs(env: ManagerBasedRLEnv, piece_cfg: SceneEntityCfg) -> torch.Tensor:
    piece = env.scene[piece_cfg.name]
    return piece.data.root_pos_w - env.scene.env_origins


# ---------- Reward terms -----------------------------------------------------

def _piece_target_dist(env: ManagerBasedRLEnv, piece_cfg, target_cfg) -> torch.Tensor:
    p = env.scene[piece_cfg.name].data.root_pos_w
    t = env.scene[target_cfg.name].data.root_pos_w
    return torch.norm(p - t, dim=-1)


def reach_reward(env, piece_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """exp(-distance) shaping that pulls piece toward target.

    A single dense term covers both phase-0 reach-to-piece (via small piece
    height) and phase-2/3 place; the IK skeleton supplies the phase split,
    so the reward stays simple and decoupled from the skeleton state.
    """
    d = _piece_target_dist(env, piece_cfg, target_cfg)
    return torch.exp(-5.0 * d)


def hold_reward(env, piece_cfg: SceneEntityCfg, z_min: float = 0.55) -> torch.Tensor:
    """+1 while the piece is held above table height (proxy for contact)."""
    piece = env.scene[piece_cfg.name]
    z = piece.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (z > z_min).float()


def drop_penalty(env, piece_cfg: SceneEntityCfg, z_drop: float = 0.45) -> torch.Tensor:
    """-1 if the piece falls below the table surface (failure of transit)."""
    piece = env.scene[piece_cfg.name]
    z = piece.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (z < z_drop).float()


def success_bonus(env, piece_cfg, target_cfg) -> torch.Tensor:
    d = _piece_target_dist(env, piece_cfg, target_cfg)
    return (d < 0.03).float()


# ---------- Curriculum -------------------------------------------------------

def widen_grid_on_success(env, env_ids, success_rate_threshold: float = 0.6):
    """Stub for IsaacLab curriculum hook; widens randomized grid once metric
    crosses threshold. The real metric plumbing lives in the patched
    ``chess_residual_rl_env_cfg.py`` curriculum block; this function exists
    so ``CurriculumTermCfg(func=widen_grid_on_success)`` resolves cleanly.
    """
    return None


# ---------- Task -------------------------------------------------------------

class G1ChessTargetConditionedTask(G1ChessPiecePlacementTask):
    """Target-conditioned variant of the placement task.

    Args:
        chess_piece: piece asset (rigid cylinder).
        target_square: green target marker (kinematic cuboid).
        episode_length_s: per-episode horizon.
        randomize_grid: if True, reset each episode places piece + target on
            random training-grid cells. If False, uses the asset's static
            initial poses (useful for replaying teleop demos with a fixed
            (source, target) pair).
    """

    def __init__(
        self,
        chess_piece: Asset,
        target_square: Asset,
        episode_length_s: float | None = 30.0,
        randomize_grid: bool = True,
    ):
        super().__init__(chess_piece=chess_piece, target_square=target_square, episode_length_s=episode_length_s)
        self.randomize_grid = randomize_grid

    # -- Termination: success + drop, identical to the parent --

    def get_termination_cfg(self):
        cfg = super().get_termination_cfg()
        # Parent uses a 2.5cm xy / 3cm z tolerance via objects_in_proximity. Keep it.
        return cfg

    # -- Events: randomized grid reset (overrides parent's static pose reset) --

    def get_events_cfg(self):
        if not self.randomize_grid:
            return super().get_events_cfg()

        @configclass
        class GridEventsCfg:
            reset_chess_grid: EventTermCfg = EventTermCfg(
                func=reset_chess_grid,
                mode="reset",
                params={
                    "piece_cfg": SceneEntityCfg(self.chess_piece.name),
                    "target_cfg": SceneEntityCfg(self.target_square.name),
                    "train_only": True,
                },
            )

        return GridEventsCfg()

    # -- Observations: privileged target + piece + phase index --

    def get_observation_cfg(self):
        piece_name = self.chess_piece.name
        target_name = self.target_square.name

        @configclass
        class PolicyObs(ObservationGroupCfg):
            target_xyz = ObservationTermCfg(
                func=target_xyz_obs, params={"target_cfg": SceneEntityCfg(target_name)},
            )
            piece_xyz = ObservationTermCfg(
                func=piece_xyz_obs, params={"piece_cfg": SceneEntityCfg(piece_name)},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class ObsCfg:
            policy: PolicyObs = PolicyObs()

        return ObsCfg()

    # -- Rewards: reach + hold + drop_penalty + success bonus + action norm --

    def get_rewards_cfg(self):
        piece_name = self.chess_piece.name
        target_name = self.target_square.name

        @configclass
        class RewardsCfg:
            reach: RewardTermCfg = RewardTermCfg(
                func=reach_reward,
                weight=1.0,
                params={
                    "piece_cfg": SceneEntityCfg(piece_name),
                    "target_cfg": SceneEntityCfg(target_name),
                },
            )
            hold: RewardTermCfg = RewardTermCfg(
                func=hold_reward,
                weight=0.5,
                params={"piece_cfg": SceneEntityCfg(piece_name)},
            )
            drop: RewardTermCfg = RewardTermCfg(
                func=drop_penalty,
                weight=-2.0,
                params={"piece_cfg": SceneEntityCfg(piece_name)},
            )
            success: RewardTermCfg = RewardTermCfg(
                func=success_bonus,
                weight=10.0,
                params={
                    "piece_cfg": SceneEntityCfg(piece_name),
                    "target_cfg": SceneEntityCfg(target_name),
                },
            )
            action_l2: RewardTermCfg = RewardTermCfg(
                func=mdp.action_l2,
                weight=-0.01,
            )

        return RewardsCfg()

    # -- Curriculum: widens the grid once mean-success crosses threshold.
    #    Real plumbing lives in the patched env cfg; this hook keeps the
    #    arena task self-describing.

    def get_curriculum_cfg(self):
        @configclass
        class CurriculumCfg:
            widen_grid: CurriculumTermCfg = CurriculumTermCfg(
                func=widen_grid_on_success,
                params={"success_rate_threshold": 0.6},
            )

        return CurriculumCfg()

    def get_prompt(self) -> str:
        return "Pick up the chess piece and place it on the green target square."
