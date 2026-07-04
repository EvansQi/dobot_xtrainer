"""Data collection engine – decoupled from teleop and main loop.

Replaces the inline cv2.imwrite + save_frame + mk_dir logic scattered
across run_control.py lines 438-456.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from data_collector.writer import AsyncImageWriter
from scripts.format_obs import save_frame
from scripts.function_util import mk_dir


class CameraBuffer:
    """Per-camera double-buffer with lock for safe cross-thread access."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rgb: Optional[np.ndarray] = None
        self.ts: float = 0.0


class DataCollector:
    """Manages camera threads, image writes, and observation saving.

    Designed to be started once and then driven by session-state changes.
    The main loop calls :meth:`tick` every iteration; the collector
    handles saving when recording is active.

    Usage
    -----
        collector = DataCollector(save_dir, cameras, compress=True)
        collector.start_camera_threads()
        ...
        while True:
            collector.tick(action, obs, session_state)
    """

    def __init__(
        self,
        save_root: str,
        project_name: str,
        cameras: Dict[str, Any],       # {name: RealSenseCamera}
        save_hz: int = 25,
        camera_fps: int = 30,
        compress: bool = True,
        jpeg_quality: int = 50,
    ):
        self._save_root = save_root
        self._project_name = project_name
        self._cameras = cameras
        self._save_hz = save_hz
        self._camera_fps = camera_fps
        self._compress = compress
        self._jpeg_quality = jpeg_quality

        # Per-camera buffers
        self._buffers: Dict[str, CameraBuffer] = {
            name: CameraBuffer() for name in cameras
        }
        self._cam_threads: Dict[str, threading.Thread] = {}
        self._cam_running = False

        # Async writer
        self._writer = AsyncImageWriter(
            num_workers=4,
            jpeg_quality=jpeg_quality,
        )

        # Current session paths (set when recording starts)
        self._session_dir: str = ""
        self._left_img_dir: str = ""
        self._right_img_dir: str = ""
        self._top_img_dir: str = ""
        self._obs_dir: str = ""

    # ── camera threads (replaces run_thread_cam) ──────────────────

    def start_camera_threads(self):
        """Launch per-camera fetch threads."""
        if self._cam_running:
            return
        self._cam_running = True
        for name, camera in self._cameras.items():
            t = threading.Thread(
                target=self._camera_worker,
                args=(name, camera),
                daemon=True,
                name=f"cam-{name}",
            )
            self._cam_threads[name] = t
            t.start()

    def stop_camera_threads(self):
        self._cam_running = False
        for t in self._cam_threads.values():
            t.join(timeout=2.0)

    def _camera_worker(self, name: str, camera):
        period = 1.0 / self._camera_fps
        buf = self._buffers[name]
        while self._cam_running:
            tic = time.perf_counter()
            try:
                rgb, _ = camera.read()
                rgb = rgb[:, :, ::-1].copy()  # BGR→RGB + deep copy
                with buf.lock:
                    buf.rgb = rgb
                    buf.ts = time.time()
            except Exception:
                pass
            elapsed = time.perf_counter() - tic
            if elapsed < period:
                time.sleep(period - elapsed)

    # ── camera frame access ───────────────────────────────────────

    def get_frame_triplet(self) -> Dict[str, np.ndarray]:
        """Return a consistent triplet {top, left, right} under locks."""
        result = {}
        for name, buf in self._buffers.items():
            with buf.lock:
                if buf.rgb is not None:
                    result[name] = buf.rgb.copy()
        return result

    # ── session directory management ──────────────────────────────

    def setup_session_dirs(self, session_ts: str):
        """Create the directory tree for a new recording session."""
        collect_dir = os.path.join(
            self._save_root, self._project_name, "collect_data"
        )
        self._session_dir = os.path.join(collect_dir, session_ts)
        self._left_img_dir  = os.path.join(self._session_dir, "leftImg")
        self._right_img_dir = os.path.join(self._session_dir, "rightImg")
        self._top_img_dir   = os.path.join(self._session_dir, "topImg")
        self._obs_dir       = os.path.join(self._session_dir, "observation")
        mk_dir(self._left_img_dir)
        mk_dir(self._right_img_dir)
        mk_dir(self._top_img_dir)
        mk_dir(self._obs_dir)
        self._writer.start()

    def teardown_session(self):
        self._writer.stop()

    # ── main tick – call every loop iteration ─────────────────────

    def tick(
        self,
        session_ts: str,
        frame_idx: int,
        action: np.ndarray,
        obs: Dict[str, np.ndarray],
    ):
        """Save one frame of data.  Non-blocking – images go to async writer.

        Args:
            session_ts:  e.g. "20240704153022"
            frame_idx:   0, 1, 2, ...
            action:      14-element joint action array
            obs:         observation dict from env.get_obs()
        """
        frames = self.get_frame_triplet()
        if not frames:
            return

        # Write images via async queue
        idx_str = str(frame_idx)
        for cam_name, img in frames.items():
            if cam_name == "top":
                filepath = os.path.join(self._top_img_dir, f"{idx_str}.jpg")
            elif cam_name == "left":
                filepath = os.path.join(self._left_img_dir, f"{idx_str}.jpg")
            elif cam_name == "right":
                filepath = os.path.join(self._right_img_dir, f"{idx_str}.jpg")
            else:
                continue
            self._writer.enqueue(img, filepath)

        # Write observation pickle (synchronous – tiny payload)
        save_frame(self._obs_dir, frame_idx, obs, action)

    # ── utilities ─────────────────────────────────────────────────

    @staticmethod
    def check_camera_health(cam_threads: Dict[str, threading.Thread]) -> bool:
        """Return True if all camera threads are alive."""
        for name, t in cam_threads.items():
            if not t.is_alive():
                print(f"[DataCollector] Camera thread '{name}' died!")
                return False
        return True

    @property
    def dropped_writes(self) -> int:
        return self._writer.dropped
