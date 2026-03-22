# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import objects_in_proximity
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class G1ChessPiecePlacementTask(TaskBase):

    def __init__(
        self,
        chess_piece: Asset,
        target_square: Asset,
        episode_length_s: float | None = 30.0,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.chess_piece = chess_piece
        self.target_square = target_square

    def get_scene_cfg(self):
        pass

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=objects_in_proximity,
            params={
                "object_cfg": SceneEntityCfg(self.chess_piece.name),
                "target_object_cfg": SceneEntityCfg(self.target_square.name),
                "max_x_separation": 0.025,
                "max_y_separation": 0.025,
                "max_z_separation": 0.030,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": 0.3,
                "asset_cfg": SceneEntityCfg(self.chess_piece.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        return EventsCfg(chess_piece=self.chess_piece)

    def get_prompt(self):
        return "Pick up the chess piece and place it on the green target square."

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.chess_piece,
            offset=np.array([-0.8, -0.8, 0.8]),
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class EventsCfg:
    """Configuration for chess piece placement."""

    reset_chess_piece_pose: EventTermCfg = MISSING

    def __init__(self, chess_piece: Asset):
        initial_pose = chess_piece.get_initial_pose()
        if initial_pose is not None:
            self.reset_chess_piece_pose = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(chess_piece.name),
                },
            )
        else:
            print(
                f"Chess piece {chess_piece.name} has no initial pose. Not setting reset event."
            )
            self.reset_chess_piece_pose = None
