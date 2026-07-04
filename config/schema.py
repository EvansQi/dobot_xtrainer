"""Typed configuration schema for Dobot XTrainer.

All hardcoded magic numbers from run_control.py, dobot.py, dobot_settings.ini
are replaced by these dataclasses.  Edit config/runtime.py to change values;
never scatter config strings/numbers across source files.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Arm (main hand + follower robot)
# ---------------------------------------------------------------------------

@dataclass
class MainHandConfig:
    """Dynamixel main-hand (leader) configuration for one arm.

    Can be converted to DobotRobotConfig via :meth:`to_dobot_config`
    (eliminates the need to parse dobot_settings.ini at runtime).
    """
    joint_ids: List[int] = field(default_factory=list)
    append_id: int = 0
    joint_offsets: List[float] = field(default_factory=list)
    joint_signs: List[int] = field(default_factory=list)
    start_joints: List[float] = field(default_factory=list)
    gripper_config: Tuple[int, int, int] = (8, 204, 174)
    port: str = "/dev/ttyUSB0"
    baud_rate: int = 1_000_000
    using_sensor: bool = False

    def to_dobot_config(self):
        """Convert to DobotRobotConfig (used by DobotAgent)."""
        from dobot_control.agents.dobot_agent import DobotRobotConfig
        return DobotRobotConfig(
            joint_ids=list(self.joint_ids),
            append_id=self.append_id,
            baud_rate=self.baud_rate,
            port=self.port,
            joint_offsets=list(self.joint_offsets),
            joint_signs=list(self.joint_signs),
            gripper_config=list(self.gripper_config),
            start_joints=list(self.start_joints),
            using_sensor=int(self.using_sensor),
        )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    serial: str
    position: str                     # "top" | "left_wrist" | "right_wrist"
    flip: bool = False
    width: int = 640
    height: int = 480


# ---------------------------------------------------------------------------
# Collection / training data pipeline
# ---------------------------------------------------------------------------

@dataclass
class CollectionConfig:
    root_dir: str = "/home/dobot/projects/datasets/"
    dataset_name: str = "dataset_package_test"
    save_hz: int = 25                # main-loop target frequency (Hz)
    camera_fps: int = 30             # camera capture rate (Hz)
    button_poll_hz: int = 100        # button-state poll rate (Hz)
    compress: bool = True
    jpeg_quality: int = 50


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

@dataclass
class SafetyBounds:
    """Workspace safety limits (mm, m/s, rad)."""
    # Nova 2 workspace (mm)
    x_left:  Tuple[float, float] = (-450, 290)
    x_right: Tuple[float, float] = (-290, 450)
    y_range: Tuple[float, float] = (-750, -160)
    z_left:  float = 44
    z_right: float = 42
    max_z_velocity: float = -1.0     # m/s
    joint_step_limit: float = 0.9    # rad / control-step


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 6001
