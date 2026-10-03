"""Live CNN detector. Used automatically once models/tap_cnn.pt exists."""

from collections import deque
from pathlib import Path
import time

import numpy as np
import torch

from .classifier import FINGERS, TapPrediction
from .cnn_model import TemporalTapCNN
from .config import AppConfig


class CnnTapDetector:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        checkpoint = torch.load(Path(config.cnn_model_path), map_location="cpu", weights_only=True)
        self._model = TemporalTapCNN()
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
        self._history: dict[str, deque[list[float]]] = {}
        self._last_hit: dict[str, float] = {}

    def reset(self) -> None:
        self._history.clear()

    def update(self, hand, now: float | None = None) -> list[TapPrediction]:
        now = time.monotonic() if now is None else now
        hand_id = hand.handedness
        sequence = self._history.setdefault(hand_id, deque(maxlen=self._config.cnn_window_frames))
        sequence.append([hand.fingertips[finger][1] - hand.wrist_y for finger in FINGERS])
        if len(sequence) < self._config.cnn_window_frames:
            return []
        features = torch.tensor(np.asarray(sequence).T[None, ...], dtype=torch.float32)
        with torch.no_grad():
            probabilities = torch.sigmoid(self._model(features))[0].numpy()
        predictions = []
        for finger, confidence in zip(FINGERS, probabilities):
            finger_id = f"{hand_id}:{finger}"
            if confidence >= self._config.cnn_confidence_threshold and now - self._last_hit.get(finger_id, 0) >= self._config.cooldown_seconds:
                self._last_hit[finger_id] = now
                predictions.append(TapPrediction(finger_id, float(confidence)))
        return predictions
