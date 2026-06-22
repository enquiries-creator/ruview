"""Home Assistant MQTT publisher with auto-discovery.

Publishes discovery configs so the entities appear automatically under one HA
device ("Hallway Entrance Cam"), then publishes state on change. Uses an
availability topic with LWT so HA shows the device offline if the service dies.
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None

log = logging.getLogger(__name__)

# entity_key -> (kind, friendly name, device_class or None, icon)
_BINARY_SENSORS = {
    "presence": ("occupancy", "Presence", "occupancy", None),
    "motion": ("motion", "Motion", "motion", None),
    "approaching": ("approaching", "Someone Approaching", "motion", "mdi:walk"),
    "sound_event": ("sound", "Sound Event", "sound", "mdi:ear-hearing"),
    "hallway_active": ("active", "Hallway Active", "occupancy", "mdi:home-account"),
}
_SENSORS = {
    "person_count": ("Person Count", None, "mdi:account-multiple"),
    "sound_label": ("Last Sound", None, "mdi:waveform"),
    "sound_level_db": ("Sound Level", "dB", "mdi:volume-high"),
    "approach_source": ("Approach Source", None, "mdi:map-marker-path"),
}


class HaMqttPublisher:
    def __init__(self, cfg):
        if mqtt is None:
            raise RuntimeError("paho-mqtt not installed. Run: pip install paho-mqtt")
        self.cfg = cfg
        self.base = f"{cfg.base_topic}/{cfg.node_id}"
        self.avail_topic = f"{self.base}/availability"
        self.state_topic = f"{self.base}/state"
        self._last_payload: dict[str, Any] | None = None

        self.client = mqtt.Client(client_id=f"camera_sense_{cfg.node_id}")
        if cfg.username:
            self.client.username_pw_set(cfg.username, cfg.password)
        self.client.will_set(self.avail_topic, "offline", retain=True)
        self.client.on_connect = self._on_connect

    def _on_connect(self, client, userdata, flags, rc):  # noqa: ANN001
        if rc == 0:
            log.info("MQTT connected to %s:%s", self.cfg.host, self.cfg.port)
            client.publish(self.avail_topic, "online", retain=True)
            self._publish_discovery()
        else:
            log.error("MQTT connect failed rc=%s", rc)

    def connect(self) -> None:
        self.client.connect(self.cfg.host, self.cfg.port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        try:
            self.client.publish(self.avail_topic, "offline", retain=True)
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    def _device(self) -> dict:
        return {
            "identifiers": [f"camera_sense_{self.cfg.node_id}"],
            "name": self.cfg.node_name,
            "manufacturer": "camera_sense",
            "model": "RuView-Vision (Tapo C100)",
        }

    def _publish_discovery(self) -> None:
        pfx = self.cfg.discovery_prefix
        dev = self._device()
        for key, (_dc_key, name, device_class, icon) in _BINARY_SENSORS.items():
            uid = f"{self.cfg.node_id}_{key}"
            cfg = {
                "name": name,
                "unique_id": uid,
                "state_topic": self.state_topic,
                "value_template": f"{{{{ value_json.{key} }}}}",
                "payload_on": True,
                "payload_off": False,
                "availability_topic": self.avail_topic,
                "device": dev,
            }
            if device_class in ("occupancy", "motion", "sound"):
                cfg["device_class"] = device_class
            if icon:
                cfg["icon"] = icon
            topic = f"{pfx}/binary_sensor/{uid}/config"
            self.client.publish(topic, json.dumps(cfg), retain=True)

        for key, (name, unit, icon) in _SENSORS.items():
            uid = f"{self.cfg.node_id}_{key}"
            cfg = {
                "name": name,
                "unique_id": uid,
                "state_topic": self.state_topic,
                "value_template": f"{{{{ value_json.{key} }}}}",
                "availability_topic": self.avail_topic,
                "device": dev,
            }
            if unit:
                cfg["unit_of_measurement"] = unit
            if icon:
                cfg["icon"] = icon
            topic = f"{pfx}/sensor/{uid}/config"
            self.client.publish(topic, json.dumps(cfg), retain=True)
        log.info("Published HA discovery for %d entities.",
                 len(_BINARY_SENSORS) + len(_SENSORS))

    def publish_state(self, state) -> None:
        payload = {
            "presence": bool(state.presence),
            "motion": bool(state.motion),
            "approaching": bool(state.approaching),
            "sound_event": bool(state.sound_event),
            "hallway_active": bool(state.hallway_active),
            "person_count": int(state.person_count),
            "sound_label": state.sound_label,
            "sound_level_db": round(float(state.sound_level_db), 1),
            "approach_source": state.approach_source,
        }
        if payload == self._last_payload:
            return
        self._last_payload = payload
        self.client.publish(self.state_topic, json.dumps(payload), retain=False)
