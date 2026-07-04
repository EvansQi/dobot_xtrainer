"""Data collection engine – fusion scheme.

During recording, frames are buffered in memory.  On session stop, everything
is flushed to a single HDF5 file + session_meta.json + integrity report.

No more scattered jpg/pkl files.  No more two-step post-processing.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from scripts.function_util import mk_dir


# ---------------------------------------------------------------------------
# Single-frame container
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    frame_id: int
    wall_time: float
    mono_time: float
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    qpos: Optional[np.ndarray] = None
    qvel: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Camera buffer (lock-protected, per-camera)
# ---------------------------------------------------------------------------

class CameraBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.rgb: Optional[np.ndarray] = None
        self.ts: float = 0.0


# ---------------------------------------------------------------------------
# DataCollector
# ---------------------------------------------------------------------------

class DataCollector:
    """Fusion-scheme data collector.

    Usage
    -----
        collector = DataCollector(save_root, project_name, cameras)
        collector.start_camera_threads()
        ...
        # Recording loop:
        while True:
            if session.recording:
                collector.tick(action, obs, session_state)
        ...
        collector.finalize_session()     # writes HDF5 + meta + integrity
        collector.stop_camera_threads()
    """

    def __init__(
        self,
        save_root: str,
        project_name: str,
        cameras: Dict[str, Any],
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

        # Session state
        self._frames: List[FrameData] = []
        self._frame_mono_times: List[float] = []
        self._session_ts: str = ""
        self._session_dir: str = ""
        self._train_dir: str = ""
        self._recording_start_wall: float = 0.0

        # Stats
        self._camera_names: List[str] = sorted(cameras.keys())
        self._dropped_writes: int = 0
        self._duplicate_frame_counts: Dict[str, int] = {}
        self._prev_cam_wall: Dict[str, float] = {}

    # ── camera threads ──────────────────────────────────────────────

    def start_camera_threads(self):
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

    def get_frame_triplet(self) -> Dict[str, np.ndarray]:
        """Return a consistent {cam_name: image} snapshot under locks."""
        result = {}
        for name, buf in self._buffers.items():
            with buf.lock:
                if buf.rgb is not None:
                    result[name] = buf.rgb.copy()
        return result

    # ── session lifecycle ───────────────────────────────────────────

    def begin_session(self, session_ts: str):
        """Prepare for a new recording session."""
        self._frames.clear()
        self._frame_mono_times.clear()
        self._dropped_writes = 0
        self._duplicate_frame_counts.clear()
        self._prev_cam_wall.clear()

        collect_dir = os.path.join(
            self._save_root, self._project_name, "collect_data"
        )
        self._session_dir = os.path.join(collect_dir, session_ts)
        self._train_dir = os.path.join(
            self._save_root, self._project_name, "train_data"
        )
        mk_dir(self._session_dir)
        mk_dir(self._train_dir)
        self._session_ts = session_ts
        self._recording_start_wall = time.time()

    def tick(
        self,
        frame_idx: int,
        action: np.ndarray,
        obs: Dict[str, np.ndarray],
    ):
        """Record one frame.  Call from main loop every iteration when recording.

        Images are deep-copied into memory immediately; no disk I/O here.
        """
        frames = self.get_frame_triplet()
        if not frames:
            return

        wall = time.time()
        mono = time.monotonic()

        # Detect duplicate frames per camera
        for name in self._camera_names:
            if name in frames:
                prev = self._prev_cam_wall.get(name)
                if prev is not None and wall == prev:
                    self._duplicate_frame_counts[name] = (
                        self._duplicate_frame_counts.get(name, 0) + 1
                    )
                self._prev_cam_wall[name] = wall

        frame = FrameData(
            frame_id=frame_idx,
            wall_time=wall,
            mono_time=mono,
            images=frames,                        # deep copies from get_frame_triplet
            qpos=obs["joint_positions"].copy(),
            qvel=obs["joint_velocities"].copy(),
            action=action.copy(),
        )
        self._frames.append(frame)
        self._frame_mono_times.append(mono)

    def finalize_session(self):
        """Stop recording and write HDF5 + session_meta.json."""
        if not self._frames:
            print("[DataCollector] No frames to finalize.")
            return

        n_frames = len(self._frames)
        print(f"[DataCollector] Finalizing {n_frames} frames...")

        t0 = time.time()

        # ── 1. Write HDF5 ─────────────────────────────────────────
        hdf5_path = self._write_hdf5()
        print(f"[DataCollector] HDF5 written: {hdf5_path} "
              f"({time.time() - t0:.1f}s)")

        # ── 2. Write session_meta.json ────────────────────────────
        meta_path = self._write_session_meta(n_frames)
        print(f"[DataCollector] Meta written: {meta_path}")

        # ── 3. Write integrity report ─────────────────────────────
        integrity_path = self._write_integrity_report(n_frames)
        print(f"[DataCollector] Integrity report: {integrity_path}")

        print(f"[DataCollector] Total finalize: {time.time() - t0:.1f}s")

    # ── HDF5 writer ─────────────────────────────────────────────────

    def _write_hdf5(self) -> str:
        """Flush all buffered frames to a single HDF5 file."""
        import h5py

        episode_path = os.path.join(self._train_dir, f"episode_0")
        hdf5_path = episode_path + ".hdf5"

        n_frames = len(self._frames)
        if n_frames == 0:
            return hdf5_path

        # Determine camera image shape from first frame
        first_images = self._frames[0].images
        cam_names = sorted(first_images.keys())
        first_img = first_images[cam_names[0]]
        h, w, c = first_img.shape

        with h5py.File(hdf5_path, "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = self._compress
            root.attrs["total_frames"] = n_frames

            obs_grp = root.create_group("observations")
            img_grp = obs_grp.create_group("images")

            # Create datasets
            for cam_name in cam_names:
                img_grp.create_dataset(
                    cam_name, (n_frames, h, w, c), dtype="uint8",
                    chunks=(1, h, w, c),
                )

            obs_grp.create_dataset("qpos", (n_frames, 14))
            obs_grp.create_dataset("qvel", (n_frames, 14))
            root.create_dataset("action", (n_frames, 14))

            # Fill datasets frame-by-frame
            for i, frame in enumerate(self._frames):
                for cam_name in cam_names:
                    if cam_name in frame.images:
                        img_grp[cam_name][i] = frame.images[cam_name]
                if frame.qpos is not None:
                    obs_grp["qpos"][i] = frame.qpos
                if frame.qvel is not None:
                    obs_grp["qvel"][i] = frame.qvel
                if frame.action is not None:
                    root["action"][i] = frame.action

        return hdf5_path

    # ── session meta ────────────────────────────────────────────────

    def _compute_frequency_stats(self) -> dict:
        n = len(self._frame_mono_times)
        if n < 2:
            return {
                "target_hz": float(self._save_hz),
                "frame_count": n,
                "actual_hz": 0.0,
            }
        ts = np.array(self._frame_mono_times)
        dts = np.diff(ts)
        duration = float(ts[-1] - ts[0])
        return {
            "target_hz": float(self._save_hz),
            "frame_count": n,
            "duration_seconds": duration,
            "actual_hz": float((n - 1) / duration) if duration > 0 else 0.0,
            "dt_mean_seconds": float(np.mean(dts)),
            "dt_std_seconds": float(np.std(dts)),
            "dt_min_seconds": float(np.min(dts)),
            "dt_max_seconds": float(np.max(dts)),
            "dt_p99_seconds": float(np.percentile(dts, 99)),
        }

    def _write_session_meta(self, n_frames: int) -> str:
        meta = {
            "project": self._project_name,
            "session_ts": self._session_ts,
            "config": {
                "save_hz": self._save_hz,
                "camera_fps": self._camera_fps,
                "compress": self._compress,
                "jpeg_quality": self._jpeg_quality,
                "camera_names": self._camera_names,
            },
            "collection": {
                "start_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(self._recording_start_wall),
                ),
                "end_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(time.time()),
                ),
            },
            "record_frequency": self._compute_frequency_stats(),
            "image_quality": {
                "duplicate_frames_per_camera": {
                    k: int(v) for k, v in self._duplicate_frame_counts.items()
                },
                "total_duplicate_frames": int(
                    sum(self._duplicate_frame_counts.values())
                ),
                "dropped_writes": self._dropped_writes,
            },
            "output": {
                "hdf5": f"train_data/episode_0.hdf5",
                "format": "robomimic-compatible",
            },
        }
        meta_path = os.path.join(self._session_dir, "session_meta.json")
        self._atomic_write_json(meta_path, meta)
        return meta_path

    # ── integrity check ─────────────────────────────────────────────

    def _write_integrity_report(self, n_frames: int) -> str:
        errors = []
        warnings = []

        # Check frame count
        if n_frames == 0:
            errors.append("No frames recorded")

        # Check frame-id continuity
        expected_ids = list(range(n_frames))
        actual_ids = [f.frame_id for f in self._frames]
        if actual_ids != expected_ids:
            missing = set(expected_ids) - set(actual_ids)
            if missing:
                errors.append(f"Missing frame IDs: {sorted(missing)}")

        # Check image availability per camera
        for cam_name in self._camera_names:
            frames_with_img = sum(
                1 for f in self._frames if cam_name in f.images
            )
            if frames_with_img == 0:
                errors.append(f"No images from camera '{cam_name}'")
            elif frames_with_img < n_frames:
                warnings.append(
                    f"Camera '{cam_name}': {frames_with_img}/{n_frames} "
                    f"frames have images"
                )

        # Check action/qpos shape
        for i, f in enumerate(self._frames):
            if f.action is not None and f.action.shape != (14,):
                errors.append(f"Frame {i}: action shape {f.action.shape}")
            if f.qpos is not None and f.qpos.shape != (14,):
                errors.append(f"Frame {i}: qpos shape {f.qpos.shape}")

        # Frequency check
        stats = self._compute_frequency_stats()
        actual_hz = stats.get("actual_hz", 0)
        target_hz = stats.get("target_hz", self._save_hz)
        if actual_hz > 0 and actual_hz < target_hz * 0.8:
            warnings.append(
                f"Actual frequency {actual_hz:.1f} Hz is <80% of "
                f"target {target_hz} Hz"
            )

        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_dir": self._session_dir,
            "status": "PASS" if not errors else "FAIL",
            "summary": {
                "total_frames": n_frames,
                "camera_count": len(self._camera_names),
                **stats,
            },
            "issues": {"errors": errors, "warnings": warnings},
        }

        report_path = os.path.join(self._session_dir, "integrity_report.json")
        self._atomic_write_json(report_path, report)
        return report_path

    # ── utilities ───────────────────────────────────────────────────

    @staticmethod
    def _atomic_write_json(path: str, payload: dict):
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)

    @staticmethod
    def check_camera_health(cam_threads: Dict[str, threading.Thread]) -> bool:
        for name, t in cam_threads.items():
            if not t.is_alive():
                print(f"[DataCollector] Camera thread '{name}' died!")
                return False
        return True
