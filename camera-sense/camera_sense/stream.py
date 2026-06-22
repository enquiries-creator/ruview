"""RTSP ingest — one demux thread yields the latest video frame + rolling audio.

Uses PyAV (ffmpeg bindings) so a *single* connection carries both video and
audio. The decode loop runs on a background thread: video frames are kept as
"latest only" (we drop stale frames so detection never lags), while audio is
accumulated into a thread-safe rolling buffer that the audio worker drains.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import av  # PyAV
except ImportError:  # pragma: no cover - dependency surfaced at runtime
    av = None

log = logging.getLogger(__name__)


class RtspStream:
    def __init__(
        self,
        url: str,
        safe_url: str,
        rtsp_transport: str = "tcp",
        audio_sample_rate: int = 16000,
        audio_buffer_seconds: float = 8.0,
        want_audio: bool = True,
    ):
        if av is None:
            raise RuntimeError(
                "PyAV is not installed. Run: pip install av"
            )
        self.url = url
        self.safe_url = safe_url
        self.rtsp_transport = rtsp_transport
        self.audio_sr = audio_sample_rate
        self.want_audio = want_audio

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._frame_seq: int = 0

        self._audio_lock = threading.Lock()
        self._audio_buf: deque[float] = deque(
            maxlen=int(audio_buffer_seconds * audio_sample_rate)
        )

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="rtsp-demux", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ #
    def _open(self):
        options = {
            "rtsp_transport": self.rtsp_transport,
            "stimeout": "5000000",   # 5s socket timeout (microseconds)
            "max_delay": "500000",
            "fflags": "nobuffer",
        }
        return av.open(self.url, options=options, timeout=10)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                log.info("Connecting to %s", self.safe_url)
                container = self._open()
                self.connected = True
                backoff = 1.0
                log.info("Connected.")

                v_stream = next(
                    (s for s in container.streams if s.type == "video"), None
                )
                a_stream = next(
                    (s for s in container.streams if s.type == "audio"), None
                )
                if self.want_audio and a_stream is None:
                    log.warning(
                        "No audio stream found — the camera's RTSP may not "
                        "expose audio, or the wrong stream path is set."
                    )

                resampler = None
                if a_stream is not None:
                    resampler = av.audio.resampler.AudioResampler(
                        format="flt", layout="mono", rate=self.audio_sr
                    )

                for packet in container.demux():
                    if self._stop.is_set():
                        break
                    if packet.dts is None:
                        continue
                    try:
                        for frame in packet.decode():
                            if packet.stream.type == "video":
                                self._on_video(frame)
                            elif packet.stream.type == "audio" and resampler:
                                self._on_audio(frame, resampler)
                    except av.AVError:
                        continue
                container.close()
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                self.connected = False
                log.warning(
                    "Stream error (%s); reconnecting in %.0fs", exc, backoff
                )
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 30.0)
        self.connected = False

    def _on_video(self, frame) -> None:
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._latest_frame = img
            self._latest_frame_ts = time.time()
            self._frame_seq += 1

    def _on_audio(self, frame, resampler) -> None:
        out = resampler.resample(frame)
        frames = out if isinstance(out, list) else [out]
        for af in frames:
            samples = af.to_ndarray().reshape(-1).astype(np.float32)
            with self._audio_lock:
                self._audio_buf.extend(samples.tolist())

    # ------------------------------------------------------------------ #
    def latest_frame(self) -> tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._frame_seq, 0.0
            return self._latest_frame.copy(), self._frame_seq, self._latest_frame_ts

    def read_audio(self, seconds: float) -> Optional[np.ndarray]:
        n = int(seconds * self.audio_sr)
        with self._audio_lock:
            if len(self._audio_buf) < n:
                return None
            data = list(self._audio_buf)[-n:]
        return np.asarray(data, dtype=np.float32)
