import numpy as np
import time
import configparser
from dobot_control.agents.dobot_agent import DobotRobotConfig
import os
from pathlib import Path
from dataclasses import dataclass


def set_light(env, which_color, which_status):
    if which_color == "red":
        env.set_do_status([3, 0])  # yellow light off
        env.set_do_status([2, 0])  # green light off
        env.set_do_status([1, which_status])  # red light on
    elif which_color == "green":
        env.set_do_status([3, 0])  # yellow light off
        env.set_do_status([2, which_status])  # green light off
        env.set_do_status([1, 0])  # red light on
    elif which_color == "yellow":
        env.set_do_status([3, which_status])  # yellow light off
        env.set_do_status([2, 0])  # green light off
        env.set_do_status([1, 0])  # red light on

def load_ini_data_camera():
    camera_dict = {"top": None, "left": None, "right": None}
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)
    for _cam in camera_dict.keys():
        camera_dict[_cam] = ini_file.get("CAMERA", _cam)
    return camera_dict

def load_ini_data_hands():
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)

    hands_dict = {"HAND_LEFT": None, "HAND_RIGHT": None}
    for _hand in hands_dict.keys():
        hands_dict[_hand] = DobotRobotConfig(
            joint_ids=[int(i) for i in ini_file.get(_hand, "joint_ids").split(",")],
            append_id=int(ini_file.get(_hand, "append_id")),
            port=ini_file.get(_hand, "port"),
            joint_offsets=[float(i) for i in ini_file.get(_hand, "joint_offsets").split(",")],
            joint_signs=[int(i) for i in ini_file.get(_hand, "joint_signs").split(",")],
            gripper_config=[int(i) for i in ini_file.get(_hand, "gripper_config").split(",")],
            start_joints=[float(i) for i in ini_file.get(_hand, "start_joints").split(",")])
    return ini_file, hands_dict


@dataclass
class GripperConfig:
    id_name: int
    pos: tuple
    port: str


def load_ini_data_gripper():
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)

    gripper_dict = {"GRIPPER_LEFT": None, "GRIPPER_RIGHT": None}
    for _gripper in gripper_dict.keys():
        gripper_dict[_gripper] = GripperConfig(id_name=int(ini_file.get(_gripper, "id")),
                                               pos=list([int(i) for i in ini_file.get(_gripper, "pos").split(",")]),
                                               port=ini_file.get(_gripper, "port"))
    return ini_file, gripper_dict

# robot init move
def robot_pose_init(env):
    # go to the first point
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])  #
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 100)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, [1, 1])

    # go to the second point
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 0])  #
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 0])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 100)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, [1, 1])


# pose check between main hand and the follower
def pose_check(env, agent):
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]
    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)
    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                start_pos[id_mask],
                joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return 0
    else:
        print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
        if len(start_pos) == len(joints):
            return 1
        else:
            return 0


# dynamic approaching
def dynamic_approach(env, agent, flag_in):
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]
    abs_deltas = max(np.abs(start_pos - joints))
    steps = min(int(abs_deltas / 0.005), 100)
    for jnt in np.linspace(joints, start_pos, steps):
        env.step(jnt, flag_in)
    # time.sleep(0.5)


# main hand pose dev check
def obs_action_check(env, agent):
    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.6).any():
        print("Action is too big")
        # print which joints are too big
        joint_index = np.where(action - joints > 0.5)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        return 0, 0
    else:
        return 1, action


# nova2 dev joint check
def servo_action_check(action, last_action, step_len=0.1):
    if (np.abs(action - last_action) > step_len).any():
        print("Servo action dev is too big")
        joint_index = np.where(np.abs(action - last_action) > step_len)
        print(action)
        print(last_action)
        for j in joint_index[0]:
            if j != 6 and j != 13:
                print(f"Joint [{j}], leader: {action[j]}, follower: {last_action[j]}, diff: {action[j] - last_action[j]}")
                return 0

    return 1


if __name__ == "__main__":
    print(load_ini_data_camera())