"""Teleop input events – shared vocabulary for all input sources.

Replaces the undocumented what_to_do[2×3] numpy array in run_control.py
with explicit, typed button events and session commands.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Button events produced by the button-monitor thread
# ---------------------------------------------------------------------------

class ButtonEvent(Enum):
    """What a physical button press means at the session level."""
    SHORT_A_LEFT   = auto()   # left  hand short press A  → toggle lock/unlock
    SHORT_A_RIGHT  = auto()   # right hand short press A  → toggle lock/unlock
    LONG_A_LEFT    = auto()   # left  hand long  press A  → start/stop servo
    LONG_A_RIGHT   = auto()   # right hand long  press A  → start/stop servo
    B_BUTTON       = auto()   # B button                 → start/stop recording


# ---------------------------------------------------------------------------
# Arm state machine
# ---------------------------------------------------------------------------

class ArmState(Enum):
    """Per-arm control state."""
    LOCKED   = auto()   # torque off, arm is limp
    UNLOCKED = auto()   # torque on, waiting for servo command
    SERVOING = auto()   # actively following the main hand


class SessionPhase(Enum):
    """Global session phase."""
    IDLE      = auto()   # nothing active
    SERVOING  = auto()   # at least one arm is servoing
    RECORDING = auto()   # data being collected


# ---------------------------------------------------------------------------
# Per-arm runtime state
# ---------------------------------------------------------------------------

@dataclass
class ArmSessionState:
    state: ArmState = ArmState.LOCKED

    @property
    def locked(self) -> bool:
        return self.state == ArmState.LOCKED

    @property
    def servoing(self) -> bool:
        return self.state == ArmState.SERVOING


# ---------------------------------------------------------------------------
# Session-level state
# ---------------------------------------------------------------------------

class SessionState:
    """Thread-safe session state machine.

    Replaces the global variables in run_control.py:
        what_to_do   (2×3 numpy array)
        start_servo  (bool)
        last_status  (2×3 numpy array)
        keys_press_count (2×3 numpy array)
        dev_what_to_do (computed diff)
    """

    def __init__(self):
        import threading
        self._lock = threading.Lock()

        self._arms = {
            "left":  ArmSessionState(),
            "right": ArmSessionState(),
        }
        self._phase = SessionPhase.IDLE

        # Session timestamp (set when recording starts)
        self._session_ts: Optional[str] = None
        self._frame_idx: int = 0

        # Callbacks – set by SessionController to decouple session logic from I/O
        self.on_lock_changed: Optional[Callable] = None
        self.on_servo_changed: Optional[Callable] = None
        self.on_recording_changed: Optional[Callable] = None

    # ── properties (thread-safe reads) ──────────────────────────

    @property
    def phase(self) -> SessionPhase:
        with self._lock:
            return self._phase

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._phase == SessionPhase.RECORDING

    @property
    def session_ts(self) -> Optional[str]:
        with self._lock:
            return self._session_ts

    @property
    def frame_idx(self) -> int:
        with self._lock:
            return self._frame_idx

    def any_servoing(self) -> bool:
        with self._lock:
            return any(a.servoing for a in self._arms.values())

    def get_active_flag(self):
        """Return [left_active, right_active] as int array for env.step()."""
        import numpy as np
        with self._lock:
            return np.array([
                int(self._arms["left"].servoing),
                int(self._arms["right"].servoing),
            ])

    def arm_state(self, side: str) -> ArmState:
        with self._lock:
            return self._arms[side].state

    # ── button event dispatch ───────────────────────────────────

    def dispatch(self, event: ButtonEvent):
        """Route a button event to the correct handler."""
        if event == ButtonEvent.SHORT_A_LEFT:
            self._toggle_lock("left")
        elif event == ButtonEvent.SHORT_A_RIGHT:
            self._toggle_lock("right")
        elif event == ButtonEvent.LONG_A_LEFT:
            self._toggle_servo("left")
        elif event == ButtonEvent.LONG_A_RIGHT:
            self._toggle_servo("right")
        elif event == ButtonEvent.B_BUTTON:
            self._toggle_recording()

    def _toggle_lock(self, side: str):
        was_servoing = False
        with self._lock:
            arm = self._arms[side]
            if arm.locked:
                arm.state = ArmState.UNLOCKED
                locked = False
            else:
                # Locking while servoing → stop servo first + fire callback
                was_servoing = arm.servoing
                if was_servoing:
                    self._stop_servo_locked(side)
                arm.state = ArmState.LOCKED
                locked = True

        if was_servoing and self.on_servo_changed:
            self.on_servo_changed(side, False)

        if self.on_lock_changed:
            self.on_lock_changed(side, locked)

    def _toggle_servo(self, side: str):
        with self._lock:
            arm = self._arms[side]
            if arm.servoing:
                self._stop_servo_locked(side)
                servoing = False
            elif arm.state == ArmState.UNLOCKED:
                arm.state = ArmState.SERVOING
                self._phase = SessionPhase.SERVOING
                servoing = True
            else:
                return  # locked → ignore

        if self.on_servo_changed:
            self.on_servo_changed(side, servoing)

    def _toggle_recording(self):
        with self._lock:
            if self._phase == SessionPhase.RECORDING:
                self._phase = SessionPhase.SERVOING
                recording = False
            elif self._phase == SessionPhase.SERVOING:
                from datetime import datetime
                self._phase = SessionPhase.RECORDING
                self._session_ts = datetime.now().strftime("%Y%m%d%H%M%S")
                self._frame_idx = 0
                recording = True
            else:
                return  # not in a state where recording is allowed

        if self.on_recording_changed:
            self.on_recording_changed(recording)

    def _stop_servo_locked(self, side: str):
        """Stop servo for one arm (caller must hold _lock)."""
        self._arms[side].state = ArmState.UNLOCKED
        if not any(a.servoing for a in self._arms.values()):
            self._phase = SessionPhase.IDLE

    def increment_frame(self):
        with self._lock:
            self._frame_idx += 1
