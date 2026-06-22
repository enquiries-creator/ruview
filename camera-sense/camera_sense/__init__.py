"""camera_sense — RuView-style presence/activity sensing from a single IP camera.

A from-scratch alternative to the ESP32 WiFi-CSI pipeline: instead of radio
Channel State Information, it derives the same *semantic* signals (presence,
motion, "someone approaching", sound events) from a camera's RTSP video **and
audio** stream, and publishes them to Home Assistant over MQTT.

It does NOT see through walls — that is unique to RF/CSI. But the camera's
microphone hears around corners, so "someone is walking up to the door" can be
detected from rising footstep/voice loudness before the person is in frame.
"""

__version__ = "0.1.0"
