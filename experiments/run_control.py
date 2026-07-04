import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.format_obs import save_frame
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
import datetime
from pathlib import Path
import requests

# ---------------------------------------------------------------------------
# Tuning knobs – adjust these to trade latency vs CPU
# ---------------------------------------------------------------------------
TARGET_LOOP_HZ = 25          # main control-loop frequency (Hz)
CAMERA_FPS     = 30          # camera capture rate (Hz)
BUTTON_POLL_HZ = 100         # button-state poll rate (Hz)
# ---------------------------------------------------------------------------

@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = False
    save_data_path = str(Path(__file__).parent.parent.parent)+"/datasets/"
    project_name = "dataset_package_test"


# Thread button: [lock or nor, servo or not, record or not]
# 0: lock, 1: unlock
# 0: stop servo, 1: servo
# 0: stop recording, 1: recording
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])
using_sensor_protection = False
is_falling = np.array([0])


# ---------------------------------------------------------------------------
# Camera buffers – protected by a single lock so the main loop can grab a
# consistent triplet of the three views in one critical section.
# ---------------------------------------------------------------------------
img_list = [
    np.zeros((480, 640, 3), dtype=np.uint8),
    np.zeros((480, 640, 3), dtype=np.uint8),
    np.zeros((480, 640, 3), dtype=np.uint8),
]
img_lock = threading.Lock()
recording_event = threading.Event()  # set when the main loop is recording


def run_thread_cam(rs_cam, which_cam):
    """Camera capture thread – runs at a fixed rate, minimal CPU."""
    period = 1.0 / CAMERA_FPS
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while True:
        t_start = time.time()
        image_cam, _ = rs_cam.read()
        image_cam = image_cam[:, :, ::-1]            # BGR → RGB

        with img_lock:
            img_list[which_cam] = image_cam

        # Only pay the JPEG-encode cost while the main loop is recording.
        # (Kept for potential future streaming use; the recording path uses
        #  cv2.imwrite directly from img_list, so this is optional.)
        # if recording_event.is_set():
        #     _, image_ = cv2.imencode('.jpg', image_cam, encode_param)

        # Rate-limit to CAMERA_FPS (busy-wait removed).
        elapsed = time.time() - t_start
        if elapsed < period:
            time.sleep(period - elapsed)


def button_monitor_realtime(agent):
    """Button/key monitor – polls at a fixed rate instead of busy-waiting."""
    last_keys_status = np.array(([0, 0, 0], [0, 0, 0]))
    start_press_status = np.array(([0, 0], [0, 0]))
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
    period = 1.0 / BUTTON_POLL_HZ

    while not is_falling[0]:
        t_start = time.time()
        now_keys = agent.get_keys()
        dev_keys = now_keys - last_keys_status

        # button a (short press = lock/unlock, long press = servo)
        for i in range(2):
            if dev_keys[i, 0] == -1:                         # pressed
                tic = time.time()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # released
                start_press_status[i, 0] = 0
                toc = time.time()
                if toc - tic < 0.5:
                    keys_press_count[i, 0] += 1
                    if keys_press_count[i, 0] % 2 == 1:
                        what_to_do[i, 0] = 1
                        print("ButtonA: [" + str(i) + "] unlock", what_to_do)
                    else:
                        what_to_do[i, 0] = 0
                        print("ButtonA: [" + str(i) + "] lock", what_to_do)
                elif toc - tic > 1:
                    keys_press_count[i, 1] += 1
                    if keys_press_count[i, 1] % 2 == 1:
                        what_to_do[i, 1] = 1
                        print("ButtonA: [" + str(i) + "] servo")
                    else:
                        what_to_do[i, 1] = 0
                        print("ButtonA: [" + str(i) + "] stop servo")

        # button B (start/stop recording)
        for i in range(2):
            if dev_keys[i, 1] == -1:
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        keys_press_count[0, 2] += 1
                        recording_event.set()
                else:
                    what_to_do[0, 2] = 0
                    keys_press_count[0, 2] += 1
                    recording_event.clear()

        # status fall (sensor protection)
        if using_sensor_protection:
            for i in range(2):
                if now_keys[i, 2] and what_to_do[i, 0]:
                    agent.set_torque(2, True)
                    is_falling[0] = 1

        last_keys_status = now_keys

        elapsed = time.time() - t_start
        if elapsed < period:
            time.sleep(period - elapsed)


# ---------------------------------------------------------------------------
# Kinematics helpers (unchanged)
# ---------------------------------------------------------------------------

def dh_transformation_matrix(theta, d, a, alpha):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def claw_width(coef):
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    claw_wid = 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo ** 2 + 0.0028)
    return claw_wid

def forward_kinematics(q0, q1, q2, q3, q4, q5, y, r_type):
    if r_type == "Nova 2":
        dh_params = [
            (q0, 0.2234, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.280, 0),
            (q2, 0, -0.225, 0),
            (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]
    if r_type == "Nova 5":
        dh_params = [
            (q0, 0.240, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.400, 0),
            (q2, 0, -0.330, 0),
            (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]

    t = np.eye(4)
    for params in dh_params:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    t_final = np.dot(t, t_tool)
    pos = t_final[:3, 3]
    return pos


def calculate_vel_pos(action, last_action, total_time, r_type):
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])

    positions = {}
    vel = {}

    for side in ['left', 'right']:
        for paw in ['left', 'right']:
            coef = 1 if paw == 'left' else -1
            claw = claw_left if side == 'left' else claw_right
            claw *= coef

            current_fk = forward_kinematics(*action[0:6] if side == 'left' else action[7:13], claw, r_type)
            last_fk = forward_kinematics(*last_action[0:6] if side == 'left' else last_action[7:13], claw, r_type)

            positions[f'{side}_{paw}'] = current_fk
            vel[f'{side}_{paw}'] = (current_fk - last_fk) / total_time

    return positions, vel


def is_within_safe_position(position, x_range, y_range, z_min):
    return x_range[0] <= position[0] <= x_range[1] and \
           y_range[0] <= position[1] <= y_range[1] and \
           position[2] > z_min


def check_pose_protection(positions, vel, what_to_do):
    protect_err = False
    warnings = []

    delta_left_left = vel['left_left']
    delta_left_right = vel['left_right']
    delta_right_left = vel['right_left']
    delta_right_right = vel['right_right']

    positions_mm = {key: value * 1000 for key, value in positions.items()}
    x_range_left = (-450, 290)
    x_range_right = (-290, 450)
    y_range = (-750, -160)
    z_range_left = 44
    z_range_right = 42

    if what_to_do[0, 1]:
        if delta_left_left[2] < -1 or delta_left_right[2] < -1:
            warnings.append("[Warn]:The left robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_left_left: {delta_left_left[2]}")
            protect_err = True
        positions_to_check = ['left_left', 'left_right']
        x_ranges = [x_range_left, x_range_left]
        z_ranges = [z_range_left, z_range_left]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The left arm is out of the safe zone!")
            protect_err = True

    if what_to_do[1, 1]:
        if delta_right_left[2] < -1 or delta_right_right[2] < -1:
            warnings.append("[Warn]:The right robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_right_left: {delta_right_left[2]}")
            protect_err = True
        positions_to_check = ['right_left', 'right_right']
        x_ranges = [x_range_right, x_range_right]
        z_ranges = [z_range_right, z_range_right]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The right arm is out of the safe zone!")
            protect_err = True

    for warning in warnings:
        print(warning)

    return protect_err

def check_joint_safety(action):
    protect_err = False
    if not (action[2] < 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    if not (action[9] > 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    return protect_err

def get_firmware_version_satisfied(robot_ip):
    try:
        response = requests.post("http://"+robot_ip+":22000/settings/version")
        if response.status_code == 200:
            rt_version = response.text.split("{")[1].split("\n\t")[3].split(":")[1].split("\"")[1].split("-")[0].split(".")
            rt_num = "".join(rt_version)
            return 1, int(rt_num)
        else:
            print("Failed to retrieve the version webpage")
            return 0, 0
    except Exception as e:
        print(e)
        return 0, 0

def get_robot_type(robot_ip):
    response = requests.post("http://"+robot_ip+":22000/properties/controllerType")
    if response.status_code == 200:
        print("The type of robot is:", eval(response.text)["name"])
        return eval(response.text)["name"]
    else:
        print("Failed to obtain the type of robot")
        return None

def check_firmware_version():
    left_version = get_firmware_version_satisfied("192.168.5.1")
    right_version = get_firmware_version_satisfied("192.168.5.2")
    if left_version[1] < 3581 or left_version[1]>=4000:
        print("[ERROR]Left hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: %s),please check and update"%left_version[1])
        return False
    if right_version[1] < 3581 or right_version[1]>=4000:
        print("[ERROR]Right hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: {%s}),please check and update"% right_version[1])
        return False
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(args):
    save_dir = args.save_data_path + args.project_name + "/collect_data"
    mk_dir(save_dir)

    # -- camera init (three threads, rate-limited) --------------------------
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    thread_cam_top   = threading.Thread(target=run_thread_cam, args=(rs1, 0), daemon=True)
    thread_cam_left  = threading.Thread(target=run_thread_cam, args=(rs2, 1), daemon=True)
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2), daemon=True)
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

    # -- agent init ---------------------------------------------------------
    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    # -- robot init (ZMQ → Dobot) ------------------------------------------
    print("Waiting to connect the robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    print("If the robot fails to initialize successfully after 5 seconds,"
          "please check that the robot network is connected correctly and "
          "make sure TCP/IP mode is turned!")
    if check_firmware_version() == False:
        return
    robot_type = get_robot_type("192.168.5.1")
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    start_servo = False
    curr_light = "dark"
    print("robot init success....")

    # -- button thread (rate-limited) ---------------------------------------
    last_status = np.array(([0, 0, 0], [0, 0, 0]))
    thread_button = threading.Thread(target=button_monitor_realtime, args=(agent,), daemon=True)
    thread_button.start()
    print("button thread init success...")

    print("-------------------------Ok, let's start------------------------")
    idx = 0
    safe_limit = 0
    total_time = 0.04
    target_dt = 1.0 / TARGET_LOOP_HZ

    while True:
        tic = time.time()

        assert thread_cam_top.is_alive(),   "Error: please check the top camera!"
        assert thread_cam_left.is_alive(),  "Error: please check the left camera!"
        assert thread_cam_right.is_alive(), "Error: please check the right camera!"
        assert not is_falling, "sensor detection!"

        action = agent.act({})
        dev_what_to_do = what_to_do.copy() - last_status
        last_status = what_to_do.copy()

        # button A: short press → lock / unlock
        for i in range(2):
            if dev_what_to_do[i, 0] != 0:
                agent.set_torque(i, not what_to_do[i, 0])

        # button A: long press → start / stop servo
        if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
            print("dynamic approach")
            for i in range(2):
                if what_to_do[i, 1]:
                    agent.set_torque(i, True)
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
            last_action = dynamic_approach(env, agent, flag_in)
            for i in range(2):
                if what_to_do[i, 0]:
                    if what_to_do[i, 1]:
                        agent.set_torque(i, False)
            start_servo = True
            obs = env.get_obs()
            if curr_light != "green":
                curr_light = set_light(env, "yellow", 1)

        if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
            if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                set_light(env, "green", 0)

        if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
            action = agent.act({})
            err3, action = servo_action_check(action, last_action, flag_in)
            assert err3 != 0, set_light(env, "red", 1)

            # Security protection
            protect_err = [False, False]
            if safe_limit < 1:
                safe_limit = safe_limit + 1
            else:
                positions, vel = calculate_vel_pos(action, last_action, total_time, robot_type)
                protect_err[0] = check_pose_protection(positions, vel, what_to_do)
                protect_err[1] = check_joint_safety(action)
            if any(protect_err):
                set_light(env, "red", 1)
                time.sleep(1)
                exit()

            # button B: start / stop recording
            if dev_what_to_do[0, 2] == 1:
                curr_light = set_light(env, "green", 1)
            elif dev_what_to_do[0, 2] == -1:
                curr_light = set_light(env, "yellow", 1)

            if what_to_do[0, 2] == 1:
                idx += 1
                left_dir  = save_dir + f"/{dt_time[0]}/leftImg/"
                right_dir = save_dir + f"/{dt_time[0]}/rightImg/"
                top_dir   = save_dir + f"/{dt_time[0]}/topImg/"
                mk_dir(right_dir)
                mk_dir(top_dir)
                if mk_dir(left_dir):
                    idx = 0

                # Grab a *consistent* triplet under the lock.
                with img_lock:
                    top_img   = img_list[0].copy()
                    left_img  = img_list[1].copy()
                    right_img = img_list[2].copy()
                cv2.imwrite(top_dir   + f"{idx}.jpg", top_img)
                cv2.imwrite(left_dir  + f"{idx}.jpg", left_img)
                cv2.imwrite(right_dir + f"{idx}.jpg", right_img)

                obs_dir = save_dir + f"/{dt_time[0]}/observation/"
                mk_dir(obs_dir)
                save_frame(obs_dir, idx, obs, action)

            obs = env.step(action, flag_in)
            obs["joint_positions"][6]  = action[6]
            obs["joint_positions"][13] = action[13]
            last_action = action
        else:
            start_servo = False
            safe_limit = 0

        # -- image display (optional) ---------------------------------------
        if args.show_img:
            with img_lock:
                show_canvas[:, :640]             = img_list[0]
                show_canvas[:, 640:640 * 2]      = img_list[1]
                show_canvas[:, 640 * 2:640 * 3]  = img_list[2]
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        # -- fixed-rate pacing -----------------------------------------------
        elapsed = time.time() - tic
        total_time = elapsed
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main(tyro.cli(Args))
