from collections.abc import Callable
import threading

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from numpy.typing import NDArray


class Camera:
    def __init__(self, index: int = 0) -> None:
        self.cap = cv2.VideoCapture(index)
        self._latest_frame: NDArray[np.uint8] | None = None
        self._lock = threading.Lock()

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        with self._lock:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._latest_frame = frame.copy()
            return ret, frame

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        with self._lock:
            if self._latest_frame is None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._latest_frame = frame.copy()
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def release(self) -> None:
        with self._lock:
            self.cap.release()


class CameraStreamTrack(VideoStreamTrack):
    kind = "video"

    def __init__(
        self,
        camera: Camera,
        overlay_text: Callable[[], str] | None = None,
    ) -> None:
        super().__init__()
        self.camera = camera
        self.overlay_text = overlay_text

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        ret, frame = self.camera.read()
        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if self.overlay_text is not None:
            text = self.overlay_text()
            if text:
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
