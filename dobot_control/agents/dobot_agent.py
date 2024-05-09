import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from dobot_control.agents.agent import Agent
from dobot_control.robots.dynamixel import DynamixelRobot
import time
import configparser
import os


@dataclass
class DobotRobotConfig:
    joint_ids: Sequence[int]
    append_id: int
    port: str
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]
    gripper_config: Tuple[int, int, int]
    start_joints: Sequence[float]

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(self, start_joints: Optional[np.ndarray] = None) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            append_id=self.append_id,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=self.port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )

class GelloAgent(Agent):
    def __init__(
        self,
        which_hand: str,
        dobot_config: Optional[DobotRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        self.which_hand = which_hand
        self.torque_enable = True
        assert dobot_config
        self._robot = dobot_config.make_robot(start_joints=start_joints)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self._robot.get_joint_state()

    def set_torque(self, _flag = False):
        self._robot.set_torque_mode(_flag)
        self.torque_enable = _flag

    def get_keys(self):
        return self._robot.get_key_status()


def main() -> None:
    pass


if __name__ == "__main__":
    ini_file_path = os.path.dirname(__file__).replace("gello/agents", '')+"scripts/dobot_config/dobot_settings.ini"
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
    print("ssss: ", hands_dict["HAND_LEFT"].joint_ids)
    print("ssss: ", hands_dict["HAND_RIGHT"].joint_ids)
    left_agent = GelloAgent(which_hand="HAND_LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = GelloAgent(which_hand="HAND_RIGHT", dobot_config=hands_dict["HAND_RIGHT"])

    # # right_agent.set_torque(False)
    # left_agent.set_torque(False)

    while 1:
        tic = time.time()
        print(left_agent.act({}))
        # print(right_agent.act({}))
    #     # aaa = left_agent.act({})
    #     toc = time.time()
    #     # print("sssssss: ", toc-tic)
    # #     if dev_keys[0, 0] == -1:
        print(left_agent.get_keys(), right_agent.get_keys())
    # #     print(right_agent.get_keys())
    #     # wait_period(20, tic)
    #     # toc = time.time()
    #     # print("aaaaaaaaaaaa", toc-tic)