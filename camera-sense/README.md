# camera_sense — RuView-style sensing from a single IP camera

A **from-scratch, fully custom** presence / motion / "someone's approaching"
sensor built from an ordinary IP camera's **video *and* audio** RTSP stream,
publishing semantic states straight into **Home Assistant** over MQTT.

It's the camera-fed cousin of RuView's ESP32 WiFi-CSI pipeline: instead of radio
Channel State Information, it derives the same *semantic* signals from what the
camera can see **and hear**.

> **Honest scope.** A camera **cannot see through walls or in the dark** — that
> is unique to RF/WiFi-CSI and needs an ESP32. What it *can* do is everything
> within line of sight (presence, people-count, motion, visual approach) **plus**
> use the microphone to hear *around corners* — footsteps/voices getting louder
> = someone walking toward the camera, detected *before* they enter the frame.
> A single mic gives the **event + loudness trend**, not direction. Direction
> needs a second mic elsewhere (e.g. your M1 MacBook in the student area).

Tuned for the **Tapo C100** at `192.168.1.75` ("Hallway / Entrance / Stairs
Cam"), but works with any RTSP camera.

---

## What you get in Home Assistant

One device ("Hallway Entrance Cam") with these entities, auto-discovered:

| Entity | Type | Meaning |
|---|---|---|
| Presence | binary | a person is seen, or a person-like sound was just heard |
| Motion | binary | visual motion right now |
| **Someone Approaching** | binary | person box growing **or** audio getting louder |
| Sound Event | binary | a sound onset (door/footstep/voice/…) just occurred |
| Hallway Active | binary | any activity recently |
| Person Count | sensor | people currently detected |
| Last Sound | sensor | `door` / `footsteps` / `speech` / `other` |
| Sound Level | sensor | dB, for tuning + automations |
| Approach Source | sensor | `vision` / `audio` / `both` |

---

## Step 1 — Enable RTSP on the Tapo C100 (required)

The C100's stream needs a **"Camera Account"** (separate from your TP-Link
cloud login). In the **Tapo app**:

1. Open the camera → **⚙ Settings → Advanced Settings → Camera Account**.
2. Create a **username + password** (this is what goes in `config.yaml`).
3. Make sure the camera's IP is reachable from the Mac mini (it's
   `192.168.1.75` per your Device Info screen).

Your stream URLs become:

```
rtsp://<user>:<pass>@192.168.1.75:554/stream1   # 1080p
rtsp://<user>:<pass>@192.168.1.75:554/stream2   # lower-res, less CPU
```

> The C100's *built-in* sound detection only does "Baby Crying" — that's the
> camera's own feature and we don't use it. We read the **raw audio** off the
> RTSP stream and do our own door/footstep/voice detection. (If you don't hear
> audio in VLC for `stream1`, try `stream2`, and confirm the mic isn't muted in
> the Tapo app.)

## Step 2 — Install on the Mac mini M4

```bash
cd camera-sense
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Apple-Silicon notes:
- `ultralytics` brings in PyTorch with **MPS** (Metal) support — keep
  `vision.device: "mps"` for GPU person detection on the M4.
- If `panns-inference`/`torchaudio` give you trouble on first install, set
  `audio.classifier: "none"` — you still get sound onset + approach-trend
  (just no door/footstep *labels*), and can enable it later.

## Step 3 — Configure

```bash
cp config.example.yaml config.yaml
# edit config.yaml: camera user, mqtt host (your HA/Mosquitto broker IP)
export CAMERA_SENSE_RTSP_PASSWORD='your-camera-account-pass'
export CAMERA_SENSE_MQTT_PASSWORD='your-mqtt-pass'
```

## Step 4 — First run (console mode, no MQTT yet)

Set `mqtt.enabled: false` and just watch it think:

```bash
python run.py --config config.yaml
```

You'll see lines like:

```
STATE presence+approaching | people=1 sound=footsteps -38dB approach=both
```

Walk up to the camera; confirm `motion`, then `presence`, then `approaching`
fire. Tune `audio.onset_rms_threshold` to your hallway's noise floor (watch the
dB values when it's quiet vs. when a door opens).

## Step 5 — Turn on Home Assistant

Set `mqtt.enabled: true` + your broker details, restart. The entities appear
automatically under **Settings → Devices → Hallway Entrance Cam**. Build
automations like *"Someone Approaching = on → announce on the HomePod"*.

## Step 6 — Run it 24/7 (launchd)

Create `~/Library/LaunchAgents/one.cognitum.camerasense.plist` pointing at the
venv's python and `run.py` with `KeepAlive=true`, then:

```bash
launchctl load ~/Library/LaunchAgents/one.cognitum.camerasense.plist
```

---

## Architecture

```
RTSP (video+audio)  ──►  stream.py   (one demux thread; latest frame + audio ring)
                          │     │
              vision.py ◄─┘     └─► audio.py
        person/motion/approach        onset / loudness-trend / PANNs labels
                          │     │
                          └─►  fusion.py   (hysteresis → semantic states)
                                   │
                          mqtt_publisher.py ──► Home Assistant (auto-discovery)
```

Every module is < 500 lines and independently testable. Heavy ML deps
(`ultralytics`, `panns-inference`) are **optional** — the service degrades
gracefully to motion + audio-energy if they're absent.

## Roadmap (next phases)

- **Vitals** — breathing (chest-motion magnification) + heart rate (rPPG) when a
  person is visible and still.
- **Fall detection** — pose-based (aspect-ratio + vertical-velocity) via the
  existing MediaPipe pipeline in the repo.
- **Rough direction** — add the M1 MacBook as a 2nd mic; localise by which node
  hears a sound first/loudest.
- **Sleep** — overnight using the C100's IR night-vision.
