# Handoff — continue on the Mac mini M4

This project (`camera-sense/`) is a custom, camera-based alternative to RuView's
ESP32 WiFi-CSI sensing. It reads a **Tapo C100** RTSP **video + audio** stream
and publishes presence / motion / "someone approaching" / sound-event states to
**Home Assistant** over MQTT. See `README.md` for the full picture.

It must run on the **Mac mini M4** (same Wi-Fi as the camera + Home Assistant) —
a cloud session can't reach `192.168.1.75`.

## Your setup (known values)

| Thing | Value |
|---|---|
| Camera | Tapo C100, static IP `192.168.1.75`, port `554` |
| RTSP user | `3Vista2102` |
| RTSP password | **not stored here** — put it in `CAMERA_SENSE_RTSP_PASSWORD` |
| Stream | `stream2` (lower-CPU); `stream1` = 1080p |
| LAN | `192.168.1.0/24`, gateway/DNS `192.168.1.1` |
| Compute | Mac mini M4 (`vision.device: "mps"`) |
| MQTT broker | TBD — check for the Mosquitto add-on in Home Assistant |

> 🔒 Change the camera password soon — the current one is weak and was shared in
> screenshots. Update it in the Tapo app → Camera Account, then update the env var.

## Pick up the branch

```bash
git fetch origin
git checkout claude/ru-view-link-b2ttkm
git pull
cd camera-sense
```

## Steps

1. **Confirm the stream (video + audio):**
   ```bash
   brew install ffmpeg            # if needed
   ffprobe "rtsp://3Vista2102:<pass>@192.168.1.75:554/stream2"
   ```
   Expect one video stream and one audio stream. No audio? Try `stream1` and
   check the mic isn't muted in the Tapo app.

2. **Install:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   On first install, if `panns-inference`/`torchaudio` fight you, set
   `audio.classifier: "none"` for now (you still get onset + approach trend).

3. **Configure (config.yaml is git-ignored — safe for creds):**
   ```bash
   cp config.example.yaml config.yaml
   export CAMERA_SENSE_RTSP_PASSWORD='your-camera-pass'
   ```
   In `config.yaml` set `camera.username: "3Vista2102"`, `camera.ip:
   "192.168.1.75"`, and `mqtt.enabled: false` for the first run.

4. **Run in console mode and tune:**
   ```bash
   python run.py --config config.yaml
   ```
   Walk toward the camera; confirm `motion → presence → approaching` fire. Tune
   `audio.onset_rms_threshold` to your hallway's quiet-vs-noise dB.

5. **Wire into Home Assistant:**
   - Check Home Assistant → Settings → Add-ons for **Mosquitto broker**
     (install it if absent). Create an MQTT user/password.
   - Set `mqtt.enabled: true`, `mqtt.host:` = your HA box IP, fill the
     username + `CAMERA_SENSE_MQTT_PASSWORD`, restart.
   - Entities auto-appear under **Settings → Devices → Hallway Entrance Cam**.

## What to tell Claude Code on the Mac mini

> Continue the `camera-sense` project on branch `claude/ru-view-link-b2ttkm`
> (PR #1). Read `camera-sense/HANDOFF.md` and `camera-sense/README.md`, then walk
> me through steps 1–5: ffprobe the stream, install deps, run console mode and
> tune, then set up Mosquitto + MQTT into Home Assistant. Keep the camera
> password in `CAMERA_SENSE_RTSP_PASSWORD`, never in a committed file.

## Roadmap after it's live

Vitals (rPPG breathing/heart-rate), pose-based fall detection, a 2nd mic (M1
MacBook) for rough sound direction, IR night-vision sleep monitoring.
