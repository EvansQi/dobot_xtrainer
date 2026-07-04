"""Workspace safety monitor.

Replaces the inline check_pose_protection() + check_joint_safety() in
run_control.py (lines 219-283) with a standalone, testable class.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config.runtime import SAFETY


@dataclass
class SafetyReport:
    safe: bool = True
    warnings: List[str] = field(default_factory=list)
    emergency_stop: bool = False


class WorkspaceMonitor:
    """Checks joint safety and workspace limits for each action.

    Usage
    -----
        monitor = WorkspaceMonitor(robot_type="Nova 2")
        ...
        report = monitor.check(action, last_action, total_time, active_flag)
        if report.emergency_stop:
            set_light(env, "red", 1)
            exit()
    """

    def __init__(self, robot_type: str = "Nova 2"):
        self._bounds = SAFETY
        self._robot_type = robot_type
        self._last_action: Optional[np.ndarray] = None

        # DH parameters (moved from run_control.py)
        if robot_type == "Nova 2":
            self._dh_a = [0.0, -0.280, -0.225, 0.0, 0.0, 0.0]
            self._dh_d = [0.2234, 0.0, 0.0, 0.1175, 0.120, 0.088]
        elif robot_type == "Nova 5":
            self._dh_a = [0.0, -0.400, -0.330, 0.0, 0.0, 0.0]
            self._dh_d = [0.240, 0.0, 0.0, 0.135, 0.120, 0.088]
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")

    # ── public API ─────────────────────────────────────────────────

    def check(
        self,
        action: np.ndarray,            # 14-element joint action
        last_action: np.ndarray,
        total_time: float,             # control period (s)
        active_flag: np.ndarray,       # [left_active, right_active]
    ) -> SafetyReport:
        """Run all safety checks.  Returns a consolidated report."""
        report = SafetyReport()

        # 1. Joint safety (J3 must be negative for left, positive for right)
        self._check_joint_safety(action, report)

        # 2. Workspace limits (forward kinematics)
        self._check_workspace(action, last_action, total_time, active_flag, report)

        # 3. Step-size limit
        self._check_step_size(action, report)

        self._last_action = action.copy()
        return report

    # ── individual checks ──────────────────────────────────────────

    @staticmethod
    def _check_joint_safety(action: np.ndarray, report: SafetyReport):
        """J3 of left arm must be < 0; J3 of right arm must be > 0."""
        if action[2] >= 0:
            report.warnings.append(
                f"Left J3 out of safe position: {np.rad2deg(action[2]):.0f}°"
            )
            report.emergency_stop = True
        if action[9] <= 0:
            report.warnings.append(
                f"Right J3 out of safe position: {np.rad2deg(action[9]):.0f}°"
            )
            report.emergency_stop = True

    def _check_step_size(self, action: np.ndarray, report: SafetyReport):
        """Warn if any joint moved too far in one step."""
        if self._last_action is None:
            return
        limit = self._bounds.joint_step_limit
        step = np.abs(action - self._last_action)
        if step.max() > limit:
            bad = np.where(step > limit)[0]
            report.warnings.append(
                f"Joint step too large: joints={list(bad)}, "
                f"step={[f'{step[j]:.2f}' for j in bad]}"
            )
            report.emergency_stop = True

    def _check_workspace(
        self,
        action: np.ndarray,
        last_action: np.ndarray,
        total_time: float,
        active_flag: np.ndarray,
        report: SafetyReport,
    ):
        """Forward-kinematics workspace + velocity check."""
        b = self._bounds

        for side_idx, side in enumerate(["left", "right"]):
            if not active_flag[side_idx]:
                continue

            offset = 0 if side == "left" else 7
            joints = action[offset:offset + 6]

            # Forward kinematics
            pos = self._forward_kinematics(*joints)

            # Velocity check (Z-direction)
            if self._last_action is not None and total_time > 0:
                last_joints = self._last_action[offset:offset + 6]
                last_pos = self._forward_kinematics(*last_joints)
                vz = (pos[2] - last_pos[2]) / total_time
                if vz < b.max_z_velocity:
                    report.warnings.append(
                        f"{side} arm Z velocity too fast: {vz:.2f} m/s"
                    )
                    report.emergency_stop = True

            # Workspace bounds (mm)
            x_mm, y_mm, z_mm = pos[0] * 1000, pos[1] * 1000, pos[2] * 1000
            x_range = b.x_left if side == "left" else b.x_right
            z_min = b.z_left if side == "left" else b.z_right

            if not (x_range[0] <= x_mm <= x_range[1]):
                report.warnings.append(
                    f"{side} arm X={x_mm:.0f} out of range {x_range}"
                )
                report.emergency_stop = True
            if not (b.y_range[0] <= y_mm <= b.y_range[1]):
                report.warnings.append(
                    f"{side} arm Y={y_mm:.0f} out of range {b.y_range}"
                )
                report.emergency_stop = True
            if z_mm <= z_min:
                report.warnings.append(
                    f"{side} arm Z={z_mm:.0f} below minimum {z_min}"
                )
                report.emergency_stop = True

    # ── forward kinematics (from run_control.py) ───────────────────

    def _forward_kinematics(self, q0, q1, q2, q3, q4, q5):
        """DH-based forward kinematics for the arm (no tool offset)."""
        dh_params = [
            (q0,           self._dh_d[0], self._dh_a[0],  np.pi / 2),
            (q1 - np.pi/2, self._dh_d[1], self._dh_a[1],  0),
            (q2,           self._dh_d[2], self._dh_a[2],  0),
            (q3 - np.pi/2, self._dh_d[3], self._dh_a[3],  np.pi / 2),
            (q4,           self._dh_d[4], self._dh_a[4], -np.pi / 2),
            (q5,           self._dh_d[5], self._dh_a[5],  0),
        ]

        t = np.eye(4)
        for theta, d, a, alpha in dh_params:
            t = t @ self._dh_matrix(theta, d, a, alpha)
        return t[:3, 3]

    @staticmethod
    def _dh_matrix(theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1],
        ])
