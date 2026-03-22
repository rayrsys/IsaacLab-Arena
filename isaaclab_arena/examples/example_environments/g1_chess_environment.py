# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class G1ChessBoardEnvironment(ExampleEnvironmentBase):

    name: str = "g1_chess_board"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim import spawners as sim_spawners

        from isaaclab_arena.assets.asset import Asset
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.dummy_task import DummyTask
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
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        # Fix the robot's root link in place (no walking/balance needed for chess)
        embodiment.scene_config.robot.spawn.articulation_props.fix_root_link = True

        # Position robot in front of the table
        embodiment.set_initial_pose(Pose(position_xyz=(0.5, -0.6, 0.7), rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Table in front of the robot
        office_table.set_initial_pose(
            Pose(position_xyz=(0.5, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        # Chess board on top of the table (table height ~0.5m with scale 0.7)
        chess_board.set_initial_pose(
            Pose(position_xyz=(0.5, 0.0, 0.6), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        scene = Scene(assets=[GroundPlane(), DomeLight(), office_table, chess_board])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=DummyTask(),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="g1_wbc_joint")
        parser.add_argument("--teleop_device", type=str, default=None)
