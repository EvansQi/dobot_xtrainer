# Dobot XTrainer

越疆（Dobot）双臂 AI 模仿学习训练平台 —— 通过 Dynamixel 主手遥操作采集演示数据，训练机器人策略模型。

A bimanual imitation-learning data-collection & training platform for Dobot Nova collaborative robots, using Dynamixel leader arms for teleoperation.

---

## 硬件架构 · Hardware

```
┌──────────────────────┐     ┌──────────────────────┐
│  Dynamixel 左手主手    │     │  Dynamixel 右手主手    │
│  (6-DOF + 夹爪, USB)  │     │  (6-DOF + 夹爪, USB)  │
└─────────┬────────────┘     └─────────┬────────────┘
          │ 关节角度                     │ 关节角度
          ▼                              ▼
┌─────────────────────────────────────────────────────┐
│                  run_collection.py                   │
│         (遥操状态机 + 安全检查 + 数据采集)              │
└─────────────────────┬───────────────────────────────┘
                      │ 关节指令 (ZMQ TCP)
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ Dobot Nova 左手   │    │ Dobot Nova 右手   │
│ (192.168.5.1)    │    │ (192.168.5.2)    │
│ 7-DOF + 夹爪     │    │ 7-DOF + 夹爪     │
└──────────────────┘    └──────────────────┘

相机: 3× RealSense D435 (top / left_wrist / right_wrist)
```

| 组件 | 型号 | 通信方式 |
|---|---|---|
| 主手 (Leader) | Dynamixel 舵机 (6×ID + 夹爪) | USB 串口 (1M / 2M bps) |
| 从臂 (Follower) | Dobot Nova 2 / Nova 5 | TCP/IP (端口 30003/29999/30004) |
| 夹爪 | Dobot 气动夹爪 | USB 串口 |
| 相机 | Intel RealSense D435 ×3 | USB 3.0 |

---

## 快速开始 · Quick Start

### 环境要求

- Ubuntu 20.04+ / Windows 10+
- Python 3.8+
- Dobot Nova 固件 ≥ V3.5.8.1

### 安装

```bash
git clone https://github.com/EvansQi/dobot_xtrainer.git
cd dobot_xtrainer
pip install -r requirements.txt
```

### 配置

编辑 `config/runtime.py`，修改机械臂 IP、相机序列号等：

```python
# config/runtime.py
LEFT_FOLLOWER.ip = "192.168.5.1"     # 左手 Dobot IP
RIGHT_FOLLOWER.ip = "192.168.5.2"    # 右手 Dobot IP
CAMERAS["top"].serial = "230322276936"  # 顶部相机序列号
```

### 采集数据

**第一步** — 启动 ZMQ 服务器（连接 Dobot 从臂）：

```bash
python experiments/launch_nodes.py
```

**第二步** — 启动采集程序：

```bash
# 采集模式（默认）
python experiments/run_collection.py

# 仅控制不采集
python experiments/run_collection.py --control-only

# 显示相机画面
python experiments/run_collection.py --show-img
```

**操作方式**：

| 按键 | 操作 |
|---|---|
| 短按 A | 锁定/解锁 主手舵机 |
| 长按 A (>1s) | 启动/停止 主从跟随 |
| 按 B | 开始/停止 录制 |

### 训练模型

```bash
# 使用 robomimic / R2D2 训练
cd robomimic-r2d2
python examples/train_imitation.py --dataset ../datasets/dataset_package_test/train_data/episode_0.hdf5
```

---

## 项目结构 · Project Structure

```
dobot_xtrainer/
├── config/                          # 配置中心（单点管理所有参数）
│   ├── schema.py                    #   类型化 dataclass
│   └── runtime.py                   #   运行时配置（替代 INI 文件）
│
├── teleop/                          # 遥操会话管理
│   ├── __init__.py                  #   ButtonEvent / SessionState / ArmState
│   └── session_controller.py        #   按钮轮询 + 状态机 + 回调
│
├── data_collector/                  # 数据采集引擎
│   └── collector.py                 #   内存缓冲 → HDF5 + meta + 完整性报告
│
├── safety/                          # 安全监控
│   └── workspace_monitor.py         #   关节限位 + 工作空间 + 速度检查
│
├── dobot_control/                   # 机器人控制层
│   ├── agents/                      #   主手 Agent (DobotAgent)
│   ├── robots/                      #   从臂控制 (DobotRobot, ZMQ)
│   ├── cameras/                     #   相机驱动 (RealSense)
│   ├── dynamixel/                   #   Dynamixel SDK 驱动
│   └── gripper/                     #   夹爪控制
│
├── experiments/                     # 入口脚本
│   ├── run_collection.py            #   ★ 主采集入口（工程化重构版）
│   ├── run_control.py               #   旧版采集入口（兼容保留）
│   ├── launch_nodes.py              #   ZMQ 服务器（Dobot 从臂连接）
│   └── run_inference.py             #   推理部署
│
├── scripts/                         # 工具脚本
│   ├── 1_find_port.py               #   自动扫描串口
│   ├── 2_get_offset.py              #   舵机零点标定
│   ├── 4_collect2train_data.py      #   旧版批处理脚本
│   ├── script_collect2train.py      #   旧版 HDF5 转换脚本
│   ├── function_util.py             #   通用工具函数
│   ├── format_obs.py                #   数据序列化
│   ├── manipulate_utils.py          #   动态逼近 / 安全检查 / 灯光
│   └── dobot_config/                #   旧版 INI 配置（已废弃，保留兼容）
│
├── robomimic-r2d2/                  # 模仿学习训练框架
├── ModelTrain/                      # 模型训练模块
├── examples/                        # API 使用示例
│   ├── example_dobot_robot.py       #   从臂控制
│   ├── example_agent.py             #   主手读取
│   ├── example_gripper.py           #   夹爪控制
│   ├── example_camera.py            #   相机采集
│   └── example_read_data_from_datasets.py  # 数据读取
│
├── third_party/                     # 第三方 SDK
│   ├── DynamixelSDK/                #   Robotis Dynamixel SDK
│   └── feetech/                     #   Feetech 舵机 SDK
│
├── config/                          # 配置层（v2 工程化重构新增）
├── teleop/                          # 遥操层（v2 新增）
├── data_collector/                  # 采集引擎（v2 新增）
├── safety/                          # 安全模块（v2 新增）
├── requirements.txt
├── version.txt
├── LICENSE
└── README.md
```

---

## 数据采集流程 · Data Collection Pipeline

### 采集 session 输出格式

```
datasets/<project_name>/
├── collect_data/
│   └── <session_ts>/                # e.g. 20240704153022
│       ├── session_meta.json        # 频率统计、重复帧、丢帧
│       └── integrity_report.json    # 完整性校验（缺帧/缺图/频率异常）
│
└── train_data/
    └── episode_0.hdf5               # robomimic 兼容 HDF5
```

### HDF5 数据结构

```
episode_0.hdf5
├── .attrs: sim=False, compress=True, total_frames=N
├── /observations/
│   ├── qpos          (N, 14) float64   # 左右臂关节位置
│   ├── qvel          (N, 14) float64   # 左右臂关节速度
│   └── images/
│       ├── top           (N, 480, 640, 3) uint8
│       ├── left_wrist     (N, 480, 640, 3) uint8
│       └── right_wrist    (N, 480, 640, 3) uint8
└── /action           (N, 14) float64   # 控制指令
```

### session_meta.json 示例

```json
{
  "project": "dataset_package_test",
  "session_ts": "20240704153022",
  "config": {
    "save_hz": 25,
    "camera_fps": 30,
    "compress": true,
    "jpeg_quality": 50
  },
  "record_frequency": {
    "target_hz": 25.0,
    "frame_count": 1250,
    "duration_seconds": 50.12,
    "actual_hz": 24.93,
    "dt_mean_seconds": 0.0401,
    "dt_std_seconds": 0.0023,
    "dt_p99_seconds": 0.0482
  },
  "image_quality": {
    "duplicate_frames_per_camera": {"top": 3, "left_wrist": 0, "right_wrist": 2},
    "total_duplicate_frames": 5,
    "dropped_writes": 0
  }
}
```

---

## API 接口 · API Reference

### 主手 (Dynamixel Agent)

```python
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.agents.agent import BimanualAgent

left  = DobotAgent(which_hand="LEFT",  dobot_config=left_config)
right = DobotAgent(which_hand="RIGHT", dobot_config=right_config)
agent = BimanualAgent(left, right)

action = agent.act({})                     # 读取 14 维关节角度
agent.set_torque(which_hand=0, _flag=True) # 左手舵机使能
keys = agent.get_keys()                    # 读取按键状态
```

### 从臂 (Dobot Follower)

```python
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.env import RobotEnv

robot = ZMQClientRobot(port=6001, host="127.0.0.1")
env = RobotEnv(robot)

obs = env.step(joints, flag_in)            # 发送关节指令
state = env.get_obs()                      # 读取关节状态
env.set_do_status([1, 0])                  # 控制 DO 端口（灯光）
```

### 从臂直连 (不使用 ZMQ)

```python
from dobot_control.robots.dobot import DobotRobot

dobot = DobotRobot("192.168.5.1", no_gripper=False)
joints = dobot.get_joint_state()           # 读取关节角度
dobot.command_joint_state(target_joints)   # 伺服控制
```

### 相机

```python
from dobot_control.cameras.realsense_camera import RealSenseCamera

cam = RealSenseCamera(flip=False, device_id="230322276936")
rgb, depth = cam.read()
```

---

## 安全特性 · Safety

| 保护项 | 说明 | 模块 |
|---|---|---|
| 关节限位 | 左臂 J3 < 0°, 右臂 J3 > 0° | `safety/workspace_monitor.py` |
| 步长限幅 | 单步 joint step < 0.9 rad | `safety/workspace_monitor.py` |
| 工作空间 | 笛卡尔 XYZ 边界检查 | `safety/workspace_monitor.py` |
| 速度监控 | Z 方向速度 < -1 m/s 触发急停 | `safety/workspace_monitor.py` |
| 固件兼容 | 检查 Dobot 固件 ≥ V3.5.8.1 | `run_collection.py` |

---

## 配置调优 · Tuning

编辑 `config/runtime.py`：

```python
# 采集频率（Hz）— 越高延迟越低，但要求相机/网络跟上
COLLECTION.save_hz = 25

# 相机帧率（Hz）— 受 RealSense USB 带宽限制
COLLECTION.camera_fps = 30

# 按钮轮询频率（Hz）
COLLECTION.button_poll_hz = 100

# JPEG 质量（1-100，仅 HDF5 压缩时参考）
COLLECTION.jpeg_quality = 50

# 安全边界（mm）
SAFETY.x_left  = (-450, 290)
SAFETY.x_right = (-290, 450)
SAFETY.z_left  = 44
SAFETY.joint_step_limit = 0.9
```

---

## 故障排查 · Troubleshooting

| 问题 | 可能原因 | 解决方法 |
|---|---|---|
| `Camera thread died` | RealSense USB 断开 | 检查 USB 连接，重启程序 |
| `robot_is_err: True` | Dobot 报警 | 检查 Dobot 示教器错误码，清除报警 |
| `ButtonA: timeout` | 主手串口未连接 | 运行 `python scripts/1_find_port.py` 扫描端口 |
| 采集频率偏低 | 相机帧率跟不上 | 降低 `camera_fps` 或 `save_hz` |
| `integrity_report: FAIL` | 缺帧 | 检查 `session_meta.json` 中的 `actual_hz` 和 `dt_p99` |
| ZMQ 连接失败 | `launch_nodes.py` 未启动 | 先运行 `python experiments/launch_nodes.py` |

---

## 版本历史 · Version History

详见 [version.txt](version.txt)

| 版本 | 日期 | 主要变更 |
|---|---|---|
| V1.0.7 | 2024/10/08 | 固件版本检查、机器人型号识别 |
| V1.0.7 | 2024/10/10 | 支持 Nova 5、1M/2M 波特率主手 |
| V1.0.8 | 2024/12/11 | 修复 frames>3500 采集 bug |
| V2.0.0 | 2025/07 | 工程化重构：配置中心化、Session 状态机、融合输出方案 |

---

## 第三方组件 · Third-Party Components

本项目包含来自以下开源项目的组件，详见 [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES)：

- [Robotis DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK) (Python)
- [robomimic](https://github.com/ARISE-Initiative/robomimic) (imitation learning framework)
- Intel RealSense SDK (librealsense)

## 许可证 · License

详见 [LICENSE](LICENSE)
