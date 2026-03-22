# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


class LibraryObject(Object):
    """
    Base class for objects in the library which are defined in this file.
    These objects have class attributes (rather than instance attributes).
    """

    name: str
    tags: list[str]
    usd_path: str
    object_type: ObjectType = ObjectType.RIGID
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(
            name=self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=self.object_type,
            scale=self.scale,
            initial_pose=initial_pose,
            **kwargs,
        )


# TODO(peterd, 2025.11.05): Update all OV drive paths to use {ISAACLAB_NUCLEUS_DIR}
# alias prior to public release once assets are synced to S3
@register_asset
class CrackerBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "cracker_box"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class MustardBottle(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "mustard_bottle"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class SugarBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sugar_box"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class TomatoSoupCan(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "tomato_soup_can"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PowerDrill(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "power_drill"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/power_drill_physics/power_drill_physics.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Microwave(LibraryObject, Openable):
    """A microwave oven."""

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "microwave"
    tags = ["object", "openable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="fixtures", file_name="Microwave039", file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    openable_joint_name = "microjoint"
    openable_open_threshold = 0.5

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_open_threshold=self.openable_open_threshold,
        )


@register_asset
class CoffeeMachine(LibraryObject, Pressable):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "coffee_machine"
    tags = ["object", "pressable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="fixtures", file_name="CoffeeMachine108", file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    pressable_joint_name = "CoffeeMachine108_Button002_joint"
    pressedness_threshold = 0.5

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            pressable_joint_name=self.pressable_joint_name,
            pressedness_threshold=self.pressedness_threshold,
        )


@register_asset
class OfficeTable(LibraryObject):
    """
    A basic office table.
    """

    name = "office_table"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd"
    default_prim_path = "{ENV_REGEX_NS}/office_table"
    scale = (1.0, 1.0, 0.7)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueSortingBin(LibraryObject):
    """
    A blue plastic sorting bin.
    """

    name = "blue_sorting_bin"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_sorting_bin"
    scale = (4.0, 2.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueExhaustPipe(LibraryObject):
    """
    A blue exhaust pipe.
    """

    name = "blue_exhaust_pipe"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_exhaust_pipe"
    scale = (0.55, 0.55, 1.4)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BrownBox(LibraryObject):
    """
    A brown box.
    """

    name = "brown_box"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/brown_box/brown_box.usd"
    default_prim_path = "{ENV_REGEX_NS}/brown_box"
    scale = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class ChessPiece(Asset):
    """A chess piece represented as a rigid cylinder primitive.

    Cylinder dimensions: radius=0.015m (~30mm diameter), height=0.04m (~40mm tall).
    """

    name = "chess_piece"
    tags = ["object"]

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(name=self.name, tags=self.tags)
        self.prim_path = prim_path or "{ENV_REGEX_NS}/chess_piece"
        self.initial_pose = initial_pose
        self.object_type = ObjectType.RIGID

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_initial_pose(self) -> Pose | None:
        return self.initial_pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_object_cfg(self) -> dict:
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        from isaaclab.sim.spawners.shapes import CylinderCfg

        init_state = RigidObjectCfg.InitialStateCfg()
        if self.initial_pose is not None:
            init_state.pos = self.initial_pose.position_xyz
            init_state.rot = self.initial_pose.rotation_wxyz

        return {
            self.name: RigidObjectCfg(
                prim_path=self.prim_path,
                spawn=CylinderCfg(
                    radius=0.015,
                    height=0.04,
                    rigid_props=RigidBodyPropertiesCfg(disable_gravity=False),
                    mass_props=MassPropertiesCfg(mass=0.05),
                    collision_props=CollisionPropertiesCfg(),
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                ),
                init_state=init_state,
            )
        }


@register_asset
class ChessTargetSquare(Asset):
    """A target square for chess piece placement. Kinematic (immovable) green marker."""

    name = "chess_target_square"
    tags = ["object"]

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(name=self.name, tags=self.tags)
        self.prim_path = prim_path or "{ENV_REGEX_NS}/chess_target_square"
        self.initial_pose = initial_pose
        self.object_type = ObjectType.RIGID

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_initial_pose(self) -> Pose | None:
        return self.initial_pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_object_cfg(self) -> dict:
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        from isaaclab.sim.spawners.shapes import CuboidCfg

        init_state = RigidObjectCfg.InitialStateCfg()
        if self.initial_pose is not None:
            init_state.pos = self.initial_pose.position_xyz
            init_state.rot = self.initial_pose.rotation_wxyz

        return {
            self.name: RigidObjectCfg(
                prim_path=self.prim_path,
                spawn=CuboidCfg(
                    size=(0.04, 0.04, 0.005),
                    rigid_props=RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        kinematic_enabled=True,
                    ),
                    collision_props=CollisionPropertiesCfg(),
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
                ),
                init_state=init_state,
            )
        }


@register_asset
class ChessBoard(LibraryObject):
    """
    A chess board with pieces exported from Blender.
    Visual only — the table underneath provides the collision surface for chess pieces.
    """

    name = "chess_board"
    tags = ["object"]
    object_type = ObjectType.BASE
    usd_path = "/home/ray/chessboardandpieces/chessboard/quit.usdc"
    default_prim_path = "{ENV_REGEX_NS}/chess_board"
    # Blender model is ~2m across, scale to ~0.4m to fit on table
    scale = (0.3, 0.3, 0.3)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)
