from gr00t.experiment.data_config import UnitreeG1DataConfig


class UnitreeG1ChessDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.ego_view"]


class UnitreeG1ChessMultiCamDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.ego_view", "video.top_view"]


class UnitreeG1ChessRightArmDataConfig(UnitreeG1DataConfig):
    """14 DOF action space: right arm (7) + right hand (7) only.
    State also observes only right arm + right hand (14D)."""
    video_keys = ["video.ego_view"]
    state_keys = ["state.right_arm", "state.right_hand"]
    action_keys = ["action.right_arm", "action.right_hand"]
