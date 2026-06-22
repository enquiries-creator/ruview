"""Application orchestrator.

Runs the RTSP demux thread plus two worker loops (vision, audio), fuses their
results, and publishes semantic states to Home Assistant. If MQTT is disabled
it prints state changes to the console — handy for first-run tuning.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from .audio import AudioWorker
from .config import AppConfig
from .fusion import Fusion
from .stream import RtspStream
from .vision import VisionWorker

log = logging.getLogger(__name__)


class Application:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.stream = RtspStream(
            url=cfg.camera.url,
            safe_url=cfg.camera.safe_url,
            rtsp_transport=cfg.camera.rtsp_transport,
            audio_sample_rate=cfg.audio.sample_rate,
            want_audio=cfg.audio.enabled,
        )
        self.vision = VisionWorker(cfg.vision) if cfg.vision.enabled else None
        self.audio = AudioWorker(cfg.audio) if cfg.audio.enabled else None
        self.fusion = Fusion(cfg.fusion)

        self._latest_vision = None
        self._latest_audio = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._publisher = None

    # ------------------------------------------------------------------ #
    def _init_publisher(self):
        if not self.cfg.mqtt.enabled:
            log.info("MQTT disabled — printing state changes to console.")
            return None
        from .mqtt_publisher import HaMqttPublisher
        pub = HaMqttPublisher(self.cfg.mqtt)
        pub.connect()
        return pub

    def _vision_loop(self) -> None:
        period = 1.0 / max(self.cfg.vision.target_fps, 1.0)
        last_seq = -1
        i = 0
        while not self._stop.is_set():
            t0 = time.time()
            frame, seq, _ts = self.stream.latest_frame()
            if frame is not None and seq != last_seq:
                last_seq = seq
                i += 1
                if i % self.cfg.vision.detect_every == 0:
                    res = self.vision.process(frame)
                    with self._lock:
                        self._latest_vision = res
            dt = time.time() - t0
            if dt < period:
                self._stop.wait(period - dt)

    def _audio_loop(self) -> None:
        hop = self.cfg.audio.hop_seconds
        while not self._stop.is_set():
            t0 = time.time()
            wav = self.stream.read_audio(self.cfg.audio.window_seconds)
            if wav is not None:
                res = self.audio.process(wav)
                with self._lock:
                    self._latest_audio = res
            dt = time.time() - t0
            if dt < hop:
                self._stop.wait(hop - dt)

    def _publish_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                v, a = self._latest_vision, self._latest_audio
            state = self.fusion.update(v, a)
            if self._publisher is not None:
                try:
                    self._publisher.publish_state(state)
                except Exception as exc:  # noqa: BLE001
                    log.warning("publish failed: %s", exc)
            else:
                self._log_state(state)
            self._stop.wait(0.5)

    _last_logged = None

    def _log_state(self, state) -> None:
        key = (
            state.presence, state.motion, state.approaching,
            state.sound_event, state.person_count, state.sound_label,
        )
        if key == self._last_logged:
            return
        self._last_logged = key
        flags = [n for n, v in (
            ("presence", state.presence), ("motion", state.motion),
            ("approaching", state.approaching), ("sound", state.sound_event),
        ) if v]
        log.info(
            "STATE %s | people=%d sound=%s %.0fdB approach=%s",
            "+".join(flags) or "idle",
            state.person_count, state.sound_label,
            state.sound_level_db, state.approach_source,
        )

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())

        self._publisher = self._init_publisher()
        self.stream.start()

        threads = [threading.Thread(target=self._publish_loop, daemon=True)]
        if self.vision is not None:
            threads.append(threading.Thread(target=self._vision_loop, daemon=True))
        if self.audio is not None:
            threads.append(threading.Thread(target=self._audio_loop, daemon=True))
        for t in threads:
            t.start()

        log.info("camera_sense running. Ctrl-C to stop.")
        while not self._stop.is_set():
            time.sleep(0.5)

        log.info("Shutting down…")
        self.stream.stop()
        if self._publisher is not None:
            self._publisher.stop()
        for t in threads:
            t.join(timeout=2)
