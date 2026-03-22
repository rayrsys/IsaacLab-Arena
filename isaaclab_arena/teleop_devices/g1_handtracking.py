# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.g1_lower_body_standing import (
    G1LowerBodyStandingRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter import (
    G1TriHandUpperBodyRetargeterCfg,
)

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase

# G1 TriHand joint names (14 joints: 7 per hand)
G1_HAND_JOINT_NAMES = [
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
]


@register_device
class G1HandTrackingTeleopDevice(TeleopDeviceBase):
    """Teleop device for G1 hand tracking via OpenXR (Quest 3)."""

    name = "g1_handtracking"

    def __init__(
        self, sim_device: str | None = None, num_open_xr_hand_joints: int = 52, enable_visualization: bool = True
    ):
        super().__init__(sim_device=sim_device)
        self.num_open_xr_hand_joints = num_open_xr_hand_joints
        self.enable_visualization = enable_visualization

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return DevicesCfg(
            devices={
                "g1_handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyRetargeterCfg(
                            enable_visualization=self.enable_visualization,
                            num_open_xr_hand_joints=self.num_open_xr_hand_joints,
                            sim_device=self.sim_device,
                            hand_joint_names=G1_HAND_JOINT_NAMES,
                        ),
                        G1LowerBodyStandingRetargeterCfg(
                            sim_device=self.sim_device,
                        ),
                    ],
                    sim_device=self.sim_device,
                    xr_cfg=embodiment.get_xr_cfg(),
                ),
            }
        )
