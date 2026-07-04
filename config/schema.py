"""Typed configuration schema for Dobot XTrainer.

All hardcoded magic numbers from run_control.py, dobot.py, dobot_settings.ini
are replaced by these dataclasses.  Edit config/runtime.py to change values;
never scatter config strings/numbers across source files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Arm (main hand + follower robot)
# ---------------------------------------------------------------------------

@dataclass
class MainHandConfig:
    """Dynamixel main-hand (leader) configuration for one arm."""
    joint_ids: List[int] = field(default_factory=list)
    append_id: int = 0
    joint_offsets: List[float] = field(default_factory=list)
    joint_signs: List[int] = field(default_factory=list)
    start_joints: List[float] = field(default_factory=list)
    gripper_config: Tuple[int, int, int] = (8, 204, 174)
    port: str = "/dev/ttyUSB0"
    baud_rate: int = 1_000_000
    using_sensor: bool = False


@dataclass
class FollowerArmConfig:
    """Dobot Nova follower (TCP-controlled) configuration for one arm."""
    ip: str = "192.168.5.1"
    robot_number: int = 2
    tool_params: Tuple[int, int, int, int, int, int, int] = (1, 0, 0, 197, 0, 0, 0)
    speed_factor: int = 20
    acc_j: int = 20
    speed_j: int = 20
    gripper_port: str = "/dev/ttyUSB2"
    gripper_id: int = 21
    gripper_pos_range: Tuple[int, int] = (2048, 3052)


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
    video_dt: float = 0.02           # frame interval for output video (s)


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

    # Firmware version floor (3.5.8.1 → int 3581)
    min_firmware: int = 3581
    max_firmware: int = 4000


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 6001
