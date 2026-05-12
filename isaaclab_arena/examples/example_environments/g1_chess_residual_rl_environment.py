# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment builder for the target-conditioned, residual-RL chess task.

Mirrors ``g1_chess_piece_placement_environment.py`` but:

  * uses ``G1ChessResidualRLTask`` so the env reports target-conditioned
    observations and the per-episode (source, target) grid reset,
  * accepts ``--source_square`` / ``--target_square`` CLI flags for
    teleop demo collection on a fixed cell (skips randomization),
  * exposes the IK action config when ``--teleop_device`` is set to a
    hand-tracking device, matching the placement env's pattern.
"""

from __future__ import annotations

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class G1ChessResidualRLEnvironment(ExampleEnvironmentBase):

    name: str = "g1_chess_residual_rl"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim import spawners as sim_spawners
        from isaaclab.utils import configclass
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

        from isaaclab_arena.assets.asset import Asset
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.g1_chess_residual_rl_task import G1ChessResidualRLTask
        from isaaclab_arena.tasks.g1_chess_target_conditioned_task import square_to_xyz
        from isaaclab_arena.utils.pose import Pose

        class GroundPlane(Asset):
            def __init__(self):
                super().__init__(name="ground_plane")

            def get_object_cfg(self):
                return {
                    "ground_plane": AssetBaseCfg(
                        prim_path="/World/defaultGroundPlane",
                        spawn=sim_spawners.GroundPlaneCfg(),
                    )
                }

        class DomeLight(Asset):
            def __init__(self):
                super().__init__(name="dome_light")

            def get_object_cfg(self):
                return {
                    "dome_light": AssetBaseCfg(
                        prim_path="/World/DomeLight",
                        spawn=sim_spawners.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
                    )
                }

        office_table = self.asset_registry.get_asset_by_name("office_table")()
        chess_board = self.asset_registry.get_asset_by_name("chess_board")()
        chess_piece = self.asset_registry.get_asset_by_name("chess_piece")()
        chess_target_square = self.asset_registry.get_asset_by_name("chess_target_square")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        embodiment.scene_config.robot.spawn.articulation_props.fix_root_link = True
        embodiment.set_initial_pose(Pose(position_xyz=(0.5, -0.6, 0.7), rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        if args_cli.teleop_device is not None and "handtracking" in args_cli.teleop_device.lower():
            from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (
                G1_UPPER_BODY_IK_ACTION_CFG,
            )

            urdf_omniverse_path = (
                f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/"
                "unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
            )
            G1_UPPER_BODY_IK_ACTION_CFG.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)

            @configclass
            class G1PureIKActionCfg:
                upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG

            embodiment.action_config = G1PureIKActionCfg()

        office_table.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        chess_board.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.6), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        # Decide whether to randomize the grid each reset (RL training) or
        # pin both piece + target to user-specified squares (teleop / replay).
        fixed_source = getattr(args_cli, "source_square", None)
        fixed_target = getattr(args_cli, "target_square", None)
        randomize_grid = fixed_source is None and fixed_target is None

        if not randomize_grid:
            sx, sy, sz = _algebraic_to_xyz(fixed_source or "d2", square_to_xyz)
            tx, ty, tz = _algebraic_to_xyz(fixed_target or "e4", square_to_xyz)
            chess_piece.set_initial_pose(Pose(position_xyz=(sx, sy, sz + 0.02), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
            chess_target_square.set_initial_pose(Pose(position_xyz=(tx, ty, tz), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        else:
            chess_piece.set_initial_pose(Pose(position_xyz=(0.5, 0.1, 0.52), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
            chess_target_square.set_initial_pose(Pose(position_xyz=(0.5, -0.1, 0.51), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        scene = Scene(
            assets=[GroundPlane(), DomeLight(), office_table, chess_board, chess_piece, chess_target_square]
        )
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1ChessResidualRLTask(
                chess_piece=chess_piece,
                target_square=chess_target_square,
                episode_length_s=30.0,
                randomize_grid=randomize_grid,
            ),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--source_square",
            type=str,
            default=None,
            help="Algebraic source square (e.g. 'd2'); disables grid randomization when set.",
        )
        parser.add_argument(
            "--target_square",
            type=str,
            default=None,
            help="Algebraic target square (e.g. 'e4'); disables grid randomization when set.",
        )


def _algebraic_to_xyz(square: str, square_to_xyz) -> tuple[float, float, float]:
    """'e4' -> (x, y, z) on the board surface. 'a1' is file=0, rank=0."""
    if len(square) != 2:
        raise ValueError(f"Bad square notation: {square!r}")
    file_char, rank_char = square[0].lower(), square[1]
    file_idx = ord(file_char) - ord("a")
    rank_idx = int(rank_char) - 1
    if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
        raise ValueError(f"Square out of board: {square!r}")
    return square_to_xyz(file_idx, rank_idx)
