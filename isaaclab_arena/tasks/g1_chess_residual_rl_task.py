# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Residual-RL chess task: IK skeleton + small PPO residual.

The skeleton is a phase-driven Cartesian waypoint follower that, given the
current piece and target poses, computes a target wrist pose for each
phase:

    0 approach above piece       -> (piece.x, piece.y, piece.z + 0.10)
    1 descend and close fingers  -> (piece.x, piece.y, piece.z + 0.005)
    2 lift and transit           -> (target.x, target.y, target.z + 0.12)
    3 descend and open fingers   -> (target.x, target.y, target.z + 0.005)
    4 retract                    -> (target.x, target.y, target.z + 0.15)

Per the plan, residual joint deltas from the PPO policy are summed with the
skeleton's joint-position output and clipped to +/-0.087 rad (~5 deg).

The skeleton is exposed as a stateless callable so the training script can
roll it forward per env without any Isaac-Sim runtime coupling beyond the
ManagerBasedRLEnv's standard ``env.scene`` API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.tasks.g1_chess_target_conditioned_task import G1ChessTargetConditionedTask


# Residual action clip — keeps the policy from inventing a novel trajectory.
RESIDUAL_CLIP_RAD: float = 0.087


# Phase advance thresholds (Cartesian-space tolerances on the right wrist).
APPROACH_TOL: float = 0.05
DESCEND_TOL: float = 0.015
LIFT_TOL: float = 0.05
PLACE_TOL: float = 0.02


@dataclass
class SkeletonState:
    """Per-env phase index, persisted across env.step calls by the trainer.

    Lives on CPU so the trainer can serialize it cleanly; phase advances are
    cheap so this is not a hot path.
    """

    phase: torch.Tensor  # int64, shape (num_envs,)

    @classmethod
    def new(cls, num_envs: int) -> "SkeletonState":
        return cls(phase=torch.zeros(num_envs, dtype=torch.int64))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.phase.zero_()
        else:
            self.phase[env_ids.cpu()] = 0


class ChessSkeleton:
    """Phase-driven IK skeleton.

    Holds no learnable parameters. Reads piece + target poses from the env
    scene; writes joint-position targets via a callable IK solver supplied
    by the caller. Decoupling IK lets the training script reuse Pink IK
    (compat with G1_UPPER_BODY_IK_ACTION_CFG) without importing it here,
    which keeps this module import-light enough to be unit-testable.

    The skeleton outputs a target end-effector pose (xyz + grasp_closed flag)
    per env; the trainer is responsible for converting (pose, grasp) into the
    14-DOF joint-target vector using the env's action manager. This matches
    how the existing eval_gr00t_chess.py pipeline already operates.
    """

    def __init__(
        self,
        piece_cfg: SceneEntityCfg,
        target_cfg: SceneEntityCfg,
        approach_height: float = 0.10,
        transit_height: float = 0.12,
        contact_height: float = 0.005,
    ):
        self.piece_cfg = piece_cfg
        self.target_cfg = target_cfg
        self.approach_height = approach_height
        self.transit_height = transit_height
        self.contact_height = contact_height

    def query(
        self,
        env: ManagerBasedRLEnv,
        state: SkeletonState,
        wrist_xyz_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (target_wrist_xyz_w, grasp_closed) for each env.

        Also mutates ``state.phase`` in place when a phase is satisfied.
        """
        piece = env.scene[self.piece_cfg.name].data.root_pos_w
        target = env.scene[self.target_cfg.name].data.root_pos_w
        num_envs = piece.shape[0]
        device = piece.device

        approach = piece.clone()
        approach[:, 2] += self.approach_height
        descend = piece.clone()
        descend[:, 2] += self.contact_height
        transit = target.clone()
        transit[:, 2] += self.transit_height
        place = target.clone()
        place[:, 2] += self.contact_height
        retract = target.clone()
        retract[:, 2] += 0.15

        phase = state.phase.to(device)
        wp = torch.zeros_like(piece)
        grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)

        wp[phase == 0] = approach[phase == 0]
        wp[phase == 1] = descend[phase == 1]
        wp[phase == 2] = transit[phase == 2]
        wp[phase == 3] = place[phase == 3]
        wp[phase == 4] = retract[phase == 4]
        grasp[phase >= 1] = True  # closed once we begin descending
        grasp[phase == 3] = False  # release at place

        # Advance phase when wrist reached current waypoint.
        d = torch.norm(wrist_xyz_w - wp, dim=-1)
        tol = torch.full((num_envs,), APPROACH_TOL, device=device)
        tol[phase == 1] = DESCEND_TOL
        tol[phase == 2] = LIFT_TOL
        tol[phase == 3] = PLACE_TOL
        tol[phase == 4] = APPROACH_TOL
        advance = (d < tol) & (phase < 4)
        new_phase = phase + advance.long()
        state.phase = new_phase.cpu()

        return wp, grasp


def apply_residual(skeleton_action: torch.Tensor, residual_action: torch.Tensor) -> torch.Tensor:
    """skel + clip(residual, +/- RESIDUAL_CLIP_RAD)."""
    clipped = torch.clamp(residual_action, -RESIDUAL_CLIP_RAD, RESIDUAL_CLIP_RAD)
    return skeleton_action + clipped


class G1ChessResidualRLTask(G1ChessTargetConditionedTask):
    """Same MDP as the target-conditioned task; advertises the skeleton API.

    The task itself stays pure-data (configs only). The skeleton is exposed
    via ``build_skeleton(...)`` so the training/eval script can construct
    one with the same SceneEntityCfg names used by the obs/reward terms.
    """

    def build_skeleton(self) -> ChessSkeleton:
        return ChessSkeleton(
            piece_cfg=SceneEntityCfg(self.chess_piece.name),
            target_cfg=SceneEntityCfg(self.target_square.name),
        )

    def get_prompt(self) -> str:
        return "Pick up the chess piece and place it on the green target square."
