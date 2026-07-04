"""Engineering-refactored data-collection entry point.

Replaces run_control.py (~470 lines of monolithic logic) with ~120 lines
that wire together modular components:

    config/runtime.py          → single-point configuration
    teleop/session_controller  → button state machine
    data_collector/collector   → camera management + HDF5 saving
    safety/workspace_monitor   → joint + workspace safety

The control link (Dynamixel → agent → ZMQ → Dobot) is unchanged.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config.runtime import (
    CAMERAS,
    COLLECTION,
    MAIN_HANDS,
    NETWORK,
)
from data_collector.collector import DataCollector
from dobot_control.agents.agent import BimanualAgent
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from safety.workspace_monitor import WorkspaceMonitor
from scripts.manipulate_utils import (
    dynamic_approach,
    robot_pose_init,
    servo_action_check,
    set_light,
)
from teleop.session_controller import SessionController


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class Args:
    robot_port: int = NETWORK.zmq_port
    hostname: str = NETWORK.zmq_host
    show_img: bool = False
    save_dir: str = str(Path(__file__).parent.parent / "datasets")
    project: str = COLLECTION.dataset_name


# ---------------------------------------------------------------------------
# Camera init helper – uses runtime config instead of INI file
# ---------------------------------------------------------------------------

def init_cameras():
    """Create RealSenseCamera instances from runtime config."""
    cameras = {}
    for name, cfg in CAMERAS.items():
        cameras[name] = RealSenseCamera(flip=cfg.flip, device_id=cfg.serial)
    return cameras


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    shutdown = threading.Event()
    target_dt = 1.0 / COLLECTION.save_hz

    # ── 1. Cameras ─────────────────────────────────────────────────
    cameras = init_cameras()

    # ── 2. Agents (Dynamixel main hands – config-driven, no INI) ───
    left_agent  = DobotAgent(which_hand="LEFT",
                             dobot_config=MAIN_HANDS["left"].to_dobot_config())
    right_agent = DobotAgent(which_hand="RIGHT",
                             dobot_config=MAIN_HANDS["right"].to_dobot_config())
    agent = BimanualAgent(left_agent, right_agent)

    # ── 3. Follower robot (ZMQ → Dobot) ────────────────────────────
    print("[Main] Connecting to robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    curr_light = "dark"
    print("[Main] Robot ready.")

    # ── 4. Safety monitor ──────────────────────────────────────────
    safety = WorkspaceMonitor(robot_type="Nova 2")

    # ── 5. Session controller (replaces button_monitor_realtime) ───
    session = SessionController(agent, button_poll_hz=COLLECTION.button_poll_hz)

    # ── 6. Data collector (replaces inline saving) ────────────────
    collector = DataCollector(
        save_root=args.save_dir,
        project_name=args.project,
        cameras=cameras,
        save_hz=COLLECTION.save_hz,
        camera_fps=COLLECTION.camera_fps,
        compress=COLLECTION.compress,
    )

    # ── 7. Wire callbacks ──────────────────────────────────────────

    def on_servo_start(side: str, active: bool):
        nonlocal curr_light
        print(f"[Main] {side} servo start – dynamic approach")
        flag = session.state.get_active_flag()
        _ = dynamic_approach(env, agent, flag)
        if curr_light != "green":
            curr_light = set_light(env, "yellow", 1)

    def on_servo_stop(side: str, active: bool):
        if not session.state.any_servoing():
            set_light(env, "green", 0)

    def on_record_start():
        nonlocal curr_light
        ts = session.state.session_ts
        print(f"[Main] Recording started – session {ts}")
        collector.begin_session(ts)
        curr_light = set_light(env, "green", 1)

    def on_record_stop():
        nonlocal curr_light
        print("[Main] Recording stopped – finalizing HDF5 + meta...")
        collector.finalize_session()
        curr_light = set_light(env, "yellow", 1)

    session.on_servo_start  = on_servo_start
    session.on_servo_stop   = on_servo_stop
    session.on_record_start = on_record_start
    session.on_record_stop  = on_record_stop

    # ── 8. Launch threads ──────────────────────────────────────────
    collector.start_camera_threads()
    time.sleep(2)  # camera warm-up
    session.start_button_thread()
    print("[Main] Camera + button threads started.")

    # ── 9. Pre-fill display canvas ─────────────────────────────────
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)

    # ── 10. Main loop ──────────────────────────────────────────────
    last_action = np.zeros(14)
    safe_limit = 0
    print("[Main] ------------------- Ready -------------------")

    try:
        while not shutdown.is_set():
            tic = time.perf_counter()

            # Safety: check camera thread health
            if not DataCollector.check_camera_health(collector._cam_threads):
                print("[Main] Camera thread died – shutting down.")
                break

            # Read action from main hands
            action = agent.act({})

            if session.state.any_servoing():
                flag = session.state.get_active_flag()

                # Servo step check (BEFORE env.step, comparing with PREVIOUS frame)
                action_ok, action = servo_action_check(action, last_action, flag)
                if not action_ok:
                    print("[Main] Servo step rejected by safety check")

                # Workspace safety check (skip first frame)
                if safe_limit < 1:
                    safe_limit += 1
                else:
                    report = safety.check(action, last_action, target_dt, flag)
                    if report.emergency_stop:
                        for w in report.warnings:
                            print(f"[Safety] {w}")
                        set_light(env, "red", 1)
                        time.sleep(1)
                        break

                # Send command to follower robot
                obs = env.step(action, flag)
                obs["joint_positions"][6]  = action[6]
                obs["joint_positions"][13] = action[13]

                # Save data if recording
                if session.state.recording:
                    collector.tick(
                        frame_idx=session.state.frame_idx,
                        action=action,
                        obs=obs,
                    )
                    session.state.increment_frame()

                last_action = action.copy()
            else:
                safe_limit = 0

            # Optional display
            if args.show_img:
                frames = collector.get_frame_triplet()
                if "top" in frames:
                    show_canvas[:, :640] = frames["top"]
                if "left" in frames:
                    show_canvas[:, 640:640 * 2] = frames["left"]
                if "right" in frames:
                    show_canvas[:, 640 * 2:640 * 3] = frames["right"]
                cv2.imshow("0", show_canvas)
                cv2.waitKey(1)

            # Rate-limit
            elapsed = time.perf_counter() - tic
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("[Main] Interrupted.")
    finally:
        shutdown.set()
        session.stop()
        # Finalize if we were in the middle of recording
        if session.state.recording:
            collector.finalize_session()
        collector.stop_camera_threads()
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main(tyro.cli(Args))
