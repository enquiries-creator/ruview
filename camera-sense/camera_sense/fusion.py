"""Fusion — turn raw vision/audio results into stable semantic states.

Each boolean state uses a "hold" timer so it doesn't flicker: once something
fires, it stays on for N seconds after the last positive evidence. This mirrors
the hysteresis used in the RuView CSI semantic-state layer.

States exposed:
  presence            person seen OR person-like sound recently
  motion              visual motion now/recently
  approaching         person box growing OR audio loudness rising
  sound_event         a sound onset recently (with last label)
  hallway_active      any activity (presence|motion|sound) recently
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class _Hold:
    """A boolean that latches on, then releases `hold` seconds after last set."""

    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self._until = 0.0

    def set(self, now: float, on: bool) -> None:
        if on:
            self._until = now + self.hold_s

    def get(self, now: float) -> bool:
        return now < self._until


@dataclass
class SensorState:
    presence: bool = False
    motion: bool = False
    approaching: bool = False
    leaving: bool = False
    sound_event: bool = False
    hallway_active: bool = False
    person_count: int = 0
    sound_label: str = "none"
    sound_level_db: float = -120.0
    approach_source: str = "none"     # "vision", "audio", "both", "none"
    ts: float = field(default_factory=time.time)


class Fusion:
    def __init__(self, cfg):
        self.cfg = cfg
        self._presence = _Hold(cfg.presence_hold_s)
        self._motion = _Hold(cfg.motion_hold_s)
        self._approach = _Hold(cfg.approaching_hold_s)
        self._sound = _Hold(cfg.sound_event_hold_s)
        self._last_count = 0
        self._last_label = "none"

    def update(self, vision, audio) -> SensorState:
        now = time.time()

        person_now = bool(vision and vision.person_count > 0)
        motion_now = bool(vision and vision.motion)
        # A footstep/speech sound implies a person even if not yet in frame.
        person_sound = bool(
            audio and audio.sound and audio.label in ("footsteps", "speech")
        )

        self._presence.set(now, person_now or person_sound)
        self._motion.set(now, motion_now)

        vis_approach = bool(vision and vision.approaching)
        aud_approach = bool(audio and audio.approaching)
        self._approach.set(now, vis_approach or aud_approach)

        if vis_approach and aud_approach:
            src = "both"
        elif vis_approach:
            src = "vision"
        elif aud_approach:
            src = "audio"
        else:
            src = "none"

        sound_now = bool(audio and audio.sound)
        self._sound.set(now, sound_now)

        if vision and vision.person_count:
            self._last_count = vision.person_count
        elif not self._presence.get(now):
            self._last_count = 0
        if audio and audio.label not in ("none", "other"):
            self._last_label = audio.label

        presence = self._presence.get(now)
        motion = self._motion.get(now)
        sound_event = self._sound.get(now)

        return SensorState(
            presence=presence,
            motion=motion,
            approaching=self._approach.get(now),
            leaving=bool(audio and audio.leaving),
            sound_event=sound_event,
            hallway_active=presence or motion or sound_event,
            person_count=self._last_count if presence else 0,
            sound_label=self._last_label if sound_event else "none",
            sound_level_db=float(audio.level_db) if audio else -120.0,
            approach_source=src,
        )
