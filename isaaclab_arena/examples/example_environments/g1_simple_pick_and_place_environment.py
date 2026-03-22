# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class G1SimplePickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "g1_simple_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim import spawners as sim_spawners
        from isaaclab.utils import configclass
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

        from isaaclab_arena.assets.asset import Asset
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.g1_chess_piece_placement_task import G1ChessPiecePlacementTask
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
        brown_box = self.asset_registry.get_asset_by_name("brown_box")()
        chess_target_square = self.asset_registry.get_asset_by_name("chess_target_square")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        # Fix the robot's root link in place (no walking needed for tabletop pick and place)
        embodiment.scene_config.robot.spawn.articulation_props.fix_root_link = True

        # Position robot in front of the table
        embodiment.set_initial_pose(Pose(position_xyz=(0.5, -0.6, 0.7), rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # If using hand tracking, swap Arena's WBC action config for pure PINK IK.
        if args_cli.teleop_device is not None and "handtracking" in args_cli.teleop_device.lower():
            from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (
                G1_UPPER_BODY_IK_ACTION_CFG,
            )

            urdf_omniverse_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
            G1_UPPER_BODY_IK_ACTION_CFG.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)

            @configclass
            class G1PureIKActionCfg:
                upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG

            embodiment.action_config = G1PureIKActionCfg()

        # Table in front of the robot
        office_table.set_initial_pose(
            Pose(position_xyz=(0.5, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        # Brown box on the table (object to pick up)
        # Table surface is ~0.55m high (original ~0.78m * scale 0.7), box must be above this
        brown_box.set_initial_pose(
            Pose(position_xyz=(0.5, 0.1, 0.58), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        # Target location (green square, ~20cm away from box)
        chess_target_square.set_initial_pose(
            Pose(position_xyz=(0.5, -0.1, 0.56), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        scene = Scene(assets=[GroundPlane(), DomeLight(), office_table, brown_box, chess_target_square])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1ChessPiecePlacementTask(brown_box, chess_target_square, episode_length_s=30.0),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
