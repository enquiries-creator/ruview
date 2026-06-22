"""Vision worker — person detection, motion, and visual "approach".

Two always-cheap signals plus one optional ML signal:

* motion       — frame differencing (no model, always on)
* person/count — YOLO (ultralytics), optional; falls back to motion-only
* approaching  — the largest person box growing quickly = walking toward camera

Designed for the hallway/entrance framing: a person walking up the corridor
toward the door grows in the frame, which is a robust "approach" cue even
before any audio is involved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

log = logging.getLogger(__name__)


@dataclass
class VisionResult:
    person_count: int = 0
    motion: bool = False
    motion_score: float = 0.0
    approaching: bool = False
    largest_box_frac: float = 0.0   # area of biggest person box / frame area
    has_model: bool = False
    ts: float = field(default_factory=time.time)


class VisionWorker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._prev_gray = None
        self._model = None
        self._last_box_frac = 0.0
        self._last_box_ts = 0.0
        if cfg.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            log.warning(
                "ultralytics not installed — running motion-only "
                "(no person count). Install with: pip install ultralytics"
            )
            return
        try:
            self._model = YOLO(self.cfg.yolo_model)
            log.info("YOLO loaded (%s) on %s", self.cfg.yolo_model, self.cfg.device)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load YOLO (%s) — motion-only.", exc)
            self._model = None

    # ------------------------------------------------------------------ #
    def _motion(self, frame: np.ndarray) -> tuple[bool, float]:
        if cv2 is None:
            return False, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0.0
        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        score = float(np.count_nonzero(thresh)) / thresh.size
        return score >= self.cfg.motion_threshold, score

    def _detect_people(self, frame: np.ndarray) -> tuple[int, float]:
        if self._model is None:
            return 0, 0.0
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        results = self._model.predict(
            frame,
            classes=[0],          # COCO class 0 = person
            conf=self.cfg.person_conf,
            device=self.cfg.device,
            verbose=False,
        )
        count = 0
        largest = 0.0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                count += 1
                x1, y1, x2, y2 = box[:4]
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                largest = max(largest, area / frame_area)
        return count, largest

    def _update_approach(self, box_frac: float) -> bool:
        now = time.time()
        approaching = False
        if self._last_box_ts and box_frac > 0 and self._last_box_frac > 0:
            dt = now - self._last_box_ts
            if dt > 0:
                growth_per_s = (box_frac - self._last_box_frac) / dt
                # normalise by previous size so "doubling" reads consistently
                rel = growth_per_s / max(self._last_box_frac, 1e-3)
                approaching = rel >= self.cfg.approach_growth_per_s
        if box_frac > 0:
            self._last_box_frac = box_frac
            self._last_box_ts = now
        return approaching

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray) -> VisionResult:
        motion, score = self._motion(frame)
        count, box_frac = self._detect_people(frame)
        approaching = self._update_approach(box_frac)
        return VisionResult(
            person_count=count,
            motion=motion,
            motion_score=score,
            approaching=approaching,
            largest_box_frac=box_frac,
            has_model=self._model is not None,
        )
