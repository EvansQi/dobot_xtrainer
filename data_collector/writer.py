"""Async image writer – non-blocking disk I/O for collection loop.

Replaces cv2.imwrite() blocking calls in the main control loop.
Multiple writer threads drain a shared queue so the collection
loop never waits for disk.
"""

import queue
import threading
import cv2


class AsyncImageWriter:
    """Thread-safe, non-blocking image writer.

    Usage
    -----
        writer = AsyncImageWriter(jpeg_quality=50)
        writer.start()
        ...
        writer.enqueue(frame, "/path/to/frame_001.jpg")
        ...
        writer.stop()
        print(f"Dropped: {writer.dropped}")
    """

    def __init__(
        self,
        num_workers: int = 4,
        jpeg_quality: int = 95,
        max_queue: int = 400,
    ):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._num_workers = num_workers
        self._jpeg_quality = jpeg_quality
        self._workers: list = []
        self._running = False
        self._dropped: int = 0

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._dropped = 0
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker,
                daemon=True,
                name=f"img-writer-{i}",
            )
            self._workers.append(t)
            t.start()

    def stop(self, drain_timeout: float = 5.0):
        """Signal workers to finish and wait for queue to drain."""
        self._running = False
        # Push sentinel None to wake up each worker waiting on .get()
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        for t in self._workers:
            t.join(timeout=drain_timeout)
        self._workers.clear()

    # ── enqueue ───────────────────────────────────────────────────

    def enqueue(self, image, filepath: str):
        """Non-blocking enqueue.  Drops frame if queue is full."""
        try:
            self._queue.put_nowait((image, filepath))
        except queue.Full:
            self._dropped += 1

    # ── worker ────────────────────────────────────────────────────

    def _worker(self):
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:          # sentinel – shutdown
                break

            img, filepath = item
            try:
                cv2.imwrite(filepath, img, jpeg_params)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    @property
    def dropped(self) -> int:
        return self._dropped
