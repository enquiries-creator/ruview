"""Configuration loading and validation.

Loads a YAML file (see ``config.example.yaml``) and overlays environment
variables for secrets so credentials never have to live in the YAML:

    CAMERA_SENSE_RTSP_PASSWORD   -> camera.password
    CAMERA_SENSE_MQTT_PASSWORD   -> mqtt.password
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class CameraConfig:
    ip: str = "192.168.1.75"
    port: int = 554
    username: str = ""
    password: str = ""
    # Tapo: "stream1" = HD (~1080p), "stream2" = lower-res/lower-CPU.
    stream: str = "stream2"
    # Force TCP transport — far more reliable than UDP for RTSP over WiFi.
    rtsp_transport: str = "tcp"

    @property
    def url(self) -> str:
        cred = ""
        if self.username:
            cred = f"{self.username}:{self.password}@"
        return f"rtsp://{cred}{self.ip}:{self.port}/{self.stream}"

    @property
    def safe_url(self) -> str:
        """URL with the password redacted, for logging."""
        cred = f"{self.username}:***@" if self.username else ""
        return f"rtsp://{cred}{self.ip}:{self.port}/{self.stream}"


@dataclass
class VisionConfig:
    enabled: bool = True
    # detection device for YOLO: "mps" (Apple Silicon GPU), "cpu", or "cuda".
    device: str = "mps"
    # Run person detection every Nth processed frame to save CPU.
    detect_every: int = 3
    target_fps: float = 8.0
    person_conf: float = 0.40
    # Frame-difference motion: fraction of changed pixels to call it "motion".
    motion_threshold: float = 0.012
    # Approach: relative growth of the largest person box per second to flag.
    approach_growth_per_s: float = 0.15
    yolo_model: str = "yolov8n.pt"


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 16000          # we resample the stream to this
    window_seconds: float = 1.0       # classifier window
    hop_seconds: float = 0.5          # how often we evaluate
    # Onset/energy detector (always available, no ML download needed):
    onset_rms_threshold: float = 0.02
    # Approaching: rising loudness trend over this many seconds.
    trend_seconds: float = 4.0
    trend_rise_db: float = 4.0
    # Optional PANNs (AudioSet) classifier for labels (door/footsteps/speech…).
    classifier: str = "panns"         # "panns" or "none"
    classifier_device: str = "cpu"


@dataclass
class MqttConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 1883
    username: str = ""
    password: str = ""
    base_topic: str = "camera_sense"
    discovery_prefix: str = "homeassistant"
    # Stable id used in entity unique_ids and the device registry entry.
    node_id: str = "hallway_cam"
    node_name: str = "Hallway Entrance Cam"


@dataclass
class FusionConfig:
    # Seconds a state stays "on" after the last positive detection (anti-flap).
    presence_hold_s: float = 8.0
    motion_hold_s: float = 3.0
    approaching_hold_s: float = 6.0
    sound_event_hold_s: float = 4.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    log_level: str = "INFO"


def _merge(dc: Any, data: dict | None) -> None:
    """Overlay dict values onto a dataclass instance, in place."""
    if not data:
        return
    for key, value in data.items():
        if hasattr(dc, key):
            setattr(dc, key, value)


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = AppConfig()
    _merge(cfg.camera, raw.get("camera"))
    _merge(cfg.vision, raw.get("vision"))
    _merge(cfg.audio, raw.get("audio"))
    _merge(cfg.mqtt, raw.get("mqtt"))
    _merge(cfg.fusion, raw.get("fusion"))
    if "log_level" in raw:
        cfg.log_level = raw["log_level"]

    # Secrets from environment override the file.
    cfg.camera.password = os.environ.get(
        "CAMERA_SENSE_RTSP_PASSWORD", cfg.camera.password
    )
    cfg.mqtt.password = os.environ.get(
        "CAMERA_SENSE_MQTT_PASSWORD", cfg.mqtt.password
    )
    return cfg
