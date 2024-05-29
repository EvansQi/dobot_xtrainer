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

def button_monitor_realtime(agent):
    # servo
    last_keys_status = np.array(([0, 0], [0, 0]))
    start_press_status = np.array(([0, 0], [0, 0]))  # start press
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))

    while 1:
        # time.sleep(0.010)
        now_keys = agent.get_keys()
        dev_keys = now_keys - last_keys_status
        # button a
        for i in range(2):
            if dev_keys[i, 0] == -1:  # button a: start
                tic = time.time()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # button a: end
                start_press_status[i, 0] = 0
                toc = time.time()
                if toc-tic < 0.5:
                    keys_press_count[i, 0] += 1
                    # print(i, keys_press_count[i, 0], "short press", toc-tic)
                    if keys_press_count[i, 0] % 2 == 1:
                        what_to_do[i, 0] = 1
                        # log_write(__file__, "ButtonA: ["+str(i)+"] unlock")
                        print("ButtonA: [" + str(i) + "] unlock", what_to_do)
                    else:
                        what_to_do[i, 0] = 0
                        # log_write(__file__, "ButtonA: [" + str(i) + "] lock")
                        print("ButtonA: [" + str(i) + "] lock", what_to_do)

                elif toc-tic > 1:
                    keys_press_count[i, 1] += 1
                    # print(i, keys_press_count[i, 1], "long press", toc-tic)
                    if keys_press_count[i, 1] % 2 == 1:
                        what_to_do[i, 1] = 1
                        # log_write(__file__, "ButtonA: [" + str(i) + "] servo")
                        print("ButtonA: [" + str(i) + "] servo")
                    else:
                        what_to_do[i, 1] = 0
                        # log_write(__file__, "ButtonA: [" + str(i) + "] stop servo")
                        print("ButtonA: [" + str(i) + "] stop servo")

        # button B
        # more than one start servo
        for i in range(2):
            if dev_keys[i, 1] == -1:  # B button pressed
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1
                        # log_write(__file__, "ButtonB: [" + str(i) + "] recording")
                        # new recording
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        keys_press_count[0, 2] += 1
                else:
                    what_to_do[0, 2] = 0
                    keys_press_count[0, 2] += 1
                    # log_write(__file__, "ButtonB: [" + str(i) + "] stop recording")

        last_keys_status = now_keys


# Thread: camera
npy_list = np.array([np.zeros(480*640*3), np.zeros(480*640*3), np.zeros(480*640*3)])
npy_len_list = np.array([0, 0, 0])
img_list = np.array([np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3))])


def run_thread_cam(rs_cam, which_cam):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while 1:
        image_cam, _ = rs_cam.read()
        image_cam = image_cam[:, :, ::-1]
        img_list[which_cam] = image_cam
        _, image_ = cv2.imencode('.jpg', image_cam, encode_param)
        npy_list[which_cam][:len(image_)] = image_
        npy_len_list[which_cam] = len(image_)


def main(args):
    # create dataset file path
    save_dir = args.save_data_path+args.project_name+"/collect_data"
    mk_dir(save_dir)

    # camera init
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_top.start()
    show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

    # agent init
    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    # pose init
    print("Waiting to connect the robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    print("If the robot fails to initialize successfully after 5 seconds,please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    start_servo = False
    curr_light = "dark"
    print("robot init success....")

    # button status init
    last_status = np.array(([0, 0, 0], [0, 0, 0]))  # init lock
    thread_button = threading.Thread(target=button_monitor_realtime, args=(agent, ))
    thread_button.start()
    print("button thread init success...")


    print("-------------------------Ok, let's start------------------------")
    idx = 0
    total_time = 0.04
    while 1:
        tic = time.time()
        action = agent.act({})
        print(action)
        dev_what_to_do = what_to_do.copy()-last_status
        last_status = what_to_do.copy()
        # button A: short press event. lock and unlock
        for i in range(2):
            if dev_what_to_do[i, 0] != 0:
                agent.set_torque(i, not what_to_do[i, 0])

        # button A: long press event. servo or not
        if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
            # pose check between main hand and the follower
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

        if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
            action = agent.act({})
            err3, action = servo_action_check(action, last_action, flag_in)
            assert err3 != 0, set_light(env, "red", 1)

            # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
            # [Note]: Modify the protection parameters in this section carefully !
            # J2, J3 speed limit to prevent falling: 2 rad/s
            protect_err = False
            delta = np.abs(action - last_action) / total_time
            if what_to_do[0, 1]:  # The left hand is in sync
                if max(delta[1:3]) > 2:
                    print("[Warn]:The left robot speed of the joint is moving too fast!")
                    print(delta)
                    protect_err = True
                # Left arm joint angle limitations:  -150<J3<0    J4>-45  (Note: This angle needs to be converted to radians)
                if not (action[2] > -2.6 and action[2] < 0 and action[3] > -0.78):
                    print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
                    print(action)
                    print(last_action)
                    protect_err = True

            if what_to_do[1, 1]:  # The right hand is in sync
                if max(delta[8:10]) > 2:
                    print("[Warn]:The right robot speed of the joint is moving too fast!")
                    print(delta)
                    protect_err = True
                # right arm joint angle limitations:  150>J3>0    J4<45   (Note: This angle needs to be converted to radians)
                if not (action[9] < 2.6 and action[9] > 0 and action[10] < 0.78):
                    print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
                    print(action)
                    print(last_action)
                    protect_err = True

            # left arm (jaw tip position) limit:  210>x>-410  -700<Y<-210  z>47;
            # right arm (jaw tip position) limit:  410>x>-210  -700<Y<-210  z>47;
            t1 = time.time()
            pos = env.get_XYZrxryrz_state()
            print(pos)
            if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 47) and \
                    (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 47)):
                print("[Warn]:The robot arm XYZ is out of the safe position! ")
                print(pos)
                protect_err = True
            t2 = time.time()
            print("t:", t2 - t1)

            if protect_err:
                set_light(env, "red", 1)
                time.sleep(1)
                exit()
            # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

            # button B: recording or not
            if dev_what_to_do[0, 2] == 1:
                curr_light = set_light(env, "green", 1)
            elif dev_what_to_do[0, 2] == -1:
                curr_light = set_light(env, "yellow", 1)
            if what_to_do[0, 2] == 1:
                idx += 1
                left_dir = save_dir + f"/{dt_time[0]}/leftImg/"
                right_dir = save_dir + f"/{dt_time[0]}/rightImg/"
                top_dir = save_dir + f"/{dt_time[0]}/topImg/"
                mk_dir(right_dir)
                mk_dir(top_dir)
                if mk_dir(left_dir):
                    idx = 0
                cv2.imwrite(top_dir + f"{idx}.jpg", img_list[0])
                cv2.imwrite(left_dir + f"{idx}.jpg", img_list[1])
                cv2.imwrite(right_dir + f"{idx}.jpg", img_list[2])

                obs_dir = save_dir + f"/{dt_time[0]}/observation/"
                mk_dir(obs_dir)
                save_frame(obs_dir, idx, obs, action)

            obs = env.step(action, flag_in)
            obs["joint_positions"][6] = action[6]
            obs["joint_positions"][13] = action[13]
            last_action = action
        else:
            start_servo = False
            set_light(env, "green", 0)

        # img show
        if args.show_img:
            show_canvas[:, :640] = np.asarray(img_list[0], dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(img_list[1], dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(img_list[2], dtype="uint8")
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        toc = time.time()
        total_time = toc-tic
        print("total time: ", total_time)


if __name__ == "__main__":
    main(tyro.cli(Args))
