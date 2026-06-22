"""Audio worker — the "hears around the corner" sense.

Always-on (no model needed):
  * RMS loudness + onset detection ("a sound just happened")
  * loudness trend over a few seconds → approaching (rising) / leaving (falling)

Optional (PANNs AudioSet tagger, ``pip install panns-inference torchaudio``):
  * coarse labels mapped to: door, footsteps, knock, speech, other

A single microphone cannot give *direction* — only the event, its loudness,
and whether it is getting louder. Direction needs a second mic somewhere else.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# AudioSet label substrings -> our coarse category.
_LABEL_MAP = {
    "door": ["door", "slam", "knock"],
    "footsteps": ["footstep", "walk", "run"],
    "speech": ["speech", "conversation", "shout", "yell", "talk"],
    "knock": ["knock", "tap"],
}


def _categorise(labels: list[str]) -> str:
    low = [l.lower() for l in labels]
    for cat, keys in _LABEL_MAP.items():
        if any(any(k in l for k in keys) for l in low):
            return cat
    return labels[0].lower() if labels else "other"


@dataclass
class AudioResult:
    sound: bool = False               # onset / above-threshold energy
    rms: float = 0.0
    level_db: float = -120.0
    approaching: bool = False         # loudness rising
    leaving: bool = False             # loudness falling
    label: str = "none"               # door / footsteps / speech / other
    label_conf: float = 0.0
    ts: float = field(default_factory=time.time)


class _PannsClassifier:
    def __init__(self, device: str = "cpu"):
        from panns_inference import AudioTagging  # heavy import, lazy
        self._labels_module = __import__(
            "panns_inference.config", fromlist=["labels"]
        )
        self.model = AudioTagging(checkpoint_path=None, device=device)
        # PANNs CNN14 is trained at 32 kHz.
        self.sr = 32000
        try:
            import torchaudio  # noqa: F401
            self._has_torchaudio = True
        except ImportError:
            self._has_torchaudio = False

    def _to_32k(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.sr:
            return wav
        if self._has_torchaudio:
            import torch
            import torchaudio
            t = torch.from_numpy(wav).float().unsqueeze(0)
            out = torchaudio.functional.resample(t, sr, self.sr)
            return out.squeeze(0).numpy()
        # crude linear resample fallback
        n = int(len(wav) * self.sr / sr)
        return np.interp(
            np.linspace(0, len(wav), n, endpoint=False),
            np.arange(len(wav)),
            wav,
        ).astype(np.float32)

    def classify(self, wav: np.ndarray, sr: int) -> tuple[str, float]:
        x = self._to_32k(wav, sr)[None, :]
        clipwise, _ = self.model.inference(x)
        clipwise = clipwise[0]
        labels = self._labels_module.labels
        top = int(np.argmax(clipwise))
        conf = float(clipwise[top])
        # gather top-5 labels for robust categorisation
        idx = np.argsort(clipwise)[-5:][::-1]
        top_labels = [labels[i] for i in idx]
        return _categorise(top_labels), conf


class AudioWorker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._rms_hist: deque[tuple[float, float]] = deque()  # (ts, db)
        self._clf = None
        if cfg.enabled and cfg.classifier == "panns":
            self._load_classifier()

    def _load_classifier(self) -> None:
        try:
            self._clf = _PannsClassifier(self.cfg.classifier_device)
            log.info("PANNs audio classifier loaded.")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "PANNs unavailable (%s) — running energy/trend only "
                "(no sound labels). Install: pip install panns-inference "
                "torchaudio",
                exc,
            )
            self._clf = None

    @staticmethod
    def _db(rms: float) -> float:
        return 20.0 * np.log10(max(rms, 1e-7))

    def _trend(self, now: float, db: float) -> tuple[bool, bool]:
        self._rms_hist.append((now, db))
        cutoff = now - self.cfg.trend_seconds
        while self._rms_hist and self._rms_hist[0][0] < cutoff:
            self._rms_hist.popleft()
        if len(self._rms_hist) < 3:
            return False, False
        first_db = self._rms_hist[0][1]
        delta = db - first_db
        approaching = delta >= self.cfg.trend_rise_db
        leaving = delta <= -self.cfg.trend_rise_db
        return approaching, leaving

    def process(self, wav: np.ndarray) -> AudioResult:
        if wav is None or len(wav) == 0:
            return AudioResult()
        rms = float(np.sqrt(np.mean(np.square(wav))))
        db = self._db(rms)
        now = time.time()
        sound = rms >= self.cfg.onset_rms_threshold
        approaching, leaving = self._trend(now, db)

        label, conf = "none", 0.0
        if sound and self._clf is not None:
            try:
                label, conf = self._clf.classify(wav, self.cfg.sample_rate)
            except Exception as exc:  # noqa: BLE001
                log.debug("classify failed: %s", exc)

        return AudioResult(
            sound=sound,
            rms=rms,
            level_db=db,
            approaching=approaching,
            leaving=leaving,
            label=label,
            label_conf=conf,
        )
