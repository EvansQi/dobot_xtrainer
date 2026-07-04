"""Teleop session controller – bridges button events to robot hardware.

Replaces the 100+ lines of scattered button-monitor + state-machine logic
in run_control.py (what_to_do, dev_what_to_do, last_status, keys_press_count,
start_servo, safe_limit, etc.).
"""

import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from teleop import ButtonEvent, SessionState, SessionPhase, ArmState


class SessionController:
    """Orchestrates the full data-collection session lifecycle.

    Responsibilities
    ----------------
    1. Run a background thread that polls Dynamixel keys, converts raw key
       transitions into ButtonEvents, and dispatches them to SessionState.
    2. Expose callbacks so that main() can react to state changes
       (e.g. dynamic_approach on servo start, set_light on record start).
    3. Provide the active-arm flag for env.step().

    Usage
    -----
        session = SessionController(agent)
        session.on_servo_start = lambda side, flag: do_approach(env, agent, flag)
        session.start_button_thread()
        ...
        while True:
            action = agent.act({})
            if session.state.any_servoing():
                env.step(action, session.state.get_active_flag())
            session.rate_limit()
    """

    def __init__(
        self,
        agent,                         # BimanualAgent
        button_poll_hz: float = 100.0,
    ):
        self._agent = agent
        self._poll_period = 1.0 / button_poll_hz
        self.state = SessionState()

        # Raw button tracking (mirrors the old local variables)
        self._last_keys = np.array(([0, 0, 0], [0, 0, 0]))
        self._press_start = np.array(([0.0, 0.0], [0.0, 0.0]))  # per-side, per-button

        # Thread control
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ------------------------------------------------------------------
        # Callbacks – set by main() to wire in hardware actions.
        # Each receives (side: str, active: bool).
        # ------------------------------------------------------------------
        self.on_servo_start:  Optional[Callable] = None  # called BEFORE approach
        self.on_servo_stop:   Optional[Callable] = None
        self.on_record_start: Optional[Callable] = None
        self.on_record_stop:  Optional[Callable] = None
        self.on_lock_changed: Optional[Callable] = None  # (side, locked)

        # Wire internal callbacks
        self.state.on_servo_changed = self._on_servo_changed
        self.state.on_recording_changed = self._on_recording_changed
        self.state.on_lock_changed = self._on_lock_changed

    # ── lifecycle ─────────────────────────────────────────────────

    def start_button_thread(self):
        """Start the background button-polling thread."""
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._button_loop, daemon=True, name="button-monitor"
        )
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── button polling loop (replaces button_monitor_realtime) ───

    def _button_loop(self):
        """Poll Dynamixel keys at fixed rate, emit ButtonEvents on edges."""
        # Re-bind local names for speed inside tight-ish loop
        agent = self._agent
        poll_period = self._poll_period
        running = self._running
        last_keys = self._last_keys
        press_start = self._press_start
        dispatch = self.state.dispatch

        while running.is_set():
            t_start = time.perf_counter()

            now_keys = agent.get_keys()
            if not len(now_keys):
                # timeout – skip this cycle
                elapsed = time.perf_counter() - t_start
                if elapsed < poll_period:
                    time.sleep(poll_period - elapsed)
                continue

            dev_keys = now_keys - last_keys
            now = time.time()

            # --- Button A (index 0): short → lock/unlock, long → servo ---
            for i, side in enumerate(["left", "right"]):
                if dev_keys[i, 0] == -1:                     # pressed
                    press_start[i, 0] = now
                if dev_keys[i, 0] == 1 and press_start[i, 0]:  # released
                    hold = now - press_start[i, 0]
                    press_start[i, 0] = 0.0
                    if hold < 0.5:
                        # short press → lock/unlock
                        dispatch(
                            ButtonEvent.SHORT_A_LEFT if i == 0
                            else ButtonEvent.SHORT_A_RIGHT
                        )
                    elif hold > 1.0:
                        # long press → start/stop servo
                        dispatch(
                            ButtonEvent.LONG_A_LEFT if i == 0
                            else ButtonEvent.LONG_A_RIGHT
                        )

            # --- Button B (index 1): toggle recording ---
            for i in range(2):
                if dev_keys[i, 1] == -1:
                    press_start[i, 1] = now
                if dev_keys[i, 1] == 1 and press_start[i, 1]:
                    press_start[i, 1] = 0.0
                    dispatch(ButtonEvent.B_BUTTON)

            # --- Sensor fall detection ---
            # (kept as hook; original using_sensor_protection logic)
            # if now_keys[...]:
            #     is_falling[0] = 1

            last_keys = now_keys

            elapsed = time.perf_counter() - t_start
            if elapsed < poll_period:
                time.sleep(poll_period - elapsed)

    # ── internal callbacks (fire user-registered callbacks) ──────

    def _on_servo_changed(self, side: str, active: bool):
        if active and self.on_servo_start:
            self.on_servo_start(side, active)
        elif not active and self.on_servo_stop:
            self.on_servo_stop(side, active)

    def _on_recording_changed(self, recording: bool):
        if recording and self.on_record_start:
            self.on_record_start()
        elif not recording and self.on_record_stop:
            self.on_record_stop()

    def _on_lock_changed(self, side: str, locked: bool):
        """Toggle Dynamixel torque when lock state changes."""
        agent_side = 0 if side == "left" else 1
        self._agent.set_torque(agent_side, not locked)
        if self.on_lock_changed:
            self.on_lock_changed(side, locked)
