"""Single-point runtime configuration.

Edit this file (not source files) to change robot wiring, camera serials,
safety limits, collection parameters, etc.

Values are extracted from the original:
  - scripts/dobot_config/dobot_settings.ini
  - experiments/run_control.py  (hardcoded constants)
  - dobot_control/robots/dobot.py
"""

from config.schema import (
    CameraConfig,
    CollectionConfig,
    FollowerArmConfig,
    MainHandConfig,
    NetworkConfig,
    SafetyBounds,
)

# ---------------------------------------------------------------------------
# Main hands (Dynamixel leaders – read by DobotAgent)
# ---------------------------------------------------------------------------

LEFT_MAIN_HAND = MainHandConfig(
    joint_ids=[1, 2, 4, 5, 6, 7],
    append_id=3,
    joint_offsets=[6.99, 0.92, 3.16, 2.17, 3.85, 3.01],
    joint_signs=[1, 1, -1, -1, -1, 1],
    start_joints=[-1.57, 0, -1.57, 0, 1.57, 1.57],
    gripper_config=(8, 204, 174),
    port="/dev/ttyUSB0",
    baud_rate=1_000_000,
)

RIGHT_MAIN_HAND = MainHandConfig(
    joint_ids=[11, 12, 14, 15, 16, 17],
    append_id=13,
    joint_offsets=[2.38, 3.82, 3.27, 2.37, 2.3, 1.6],
    joint_signs=[1, 1, -1, -1, -1, 1],
    start_joints=[1.57, 0, 1.57, 0, -1.57, -1.57],
    gripper_config=(18, 203, 173),
    port="/dev/ttyUSB3",
    baud_rate=1_000_000,
)

MAIN_HANDS = {"left": LEFT_MAIN_HAND, "right": RIGHT_MAIN_HAND}

# ---------------------------------------------------------------------------
# Follower robots (Dobot Nova – controlled via TCP)
# ---------------------------------------------------------------------------

LEFT_FOLLOWER = FollowerArmConfig(
    ip="192.168.5.1",
    robot_number=2,
    gripper_port="/dev/ttyUSB2",
    gripper_id=21,
    gripper_pos_range=(2048, 3052),
)

RIGHT_FOLLOWER = FollowerArmConfig(
    ip="192.168.5.2",
    robot_number=2,
    gripper_port="/dev/ttyUSB1",
    gripper_id=22,
    gripper_pos_range=(2048, 3052),
)

FOLLOWERS = {"left": LEFT_FOLLOWER, "right": RIGHT_FOLLOWER}

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

CAMERAS = {
    "top":    CameraConfig(serial="230322276936", position="top", flip=True),
    "left":   CameraConfig(serial="218622275674", position="left_wrist"),
    "right":  CameraConfig(serial="218622275344", position="right_wrist", flip=True),
}

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

COLLECTION = CollectionConfig()

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

SAFETY = SafetyBounds()

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

NETWORK = NetworkConfig()
