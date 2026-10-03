"""Low-latency landmark-motion baseline for AirBeat."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .classifier import FINGERS, TapPrediction
from .config import AppConfig

if TYPE_CHECKING:
    from .hand_tracker import HandFrame


@dataclass
class FingerMotion:
    position: float
    armed: bool = True


class LandmarkTapDetector:
    """Triggers a tap once, then requires a lift before re-arming that finger.

    Each tip is measured relative to its wrist. Moving the whole hand therefore
    does not look like five simultaneous taps, which matters in DJ control mode.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._motion: dict[str, FingerMotion] = {}
        self._last_hit: dict[str, float] = {}

    def reset(self) -> None:
        self._motion.clear()

    def update(self, hand: "HandFrame", now: float | None = None) -> list[TapPrediction]:
        now = time.monotonic() if now is None else now
        predictions: list[TapPrediction] = []
        for finger, (_, y) in hand.fingertips.items():
            finger_id = f"{hand.handedness}:{finger}"
            # Normalized camera y increases downward. Subtracting the wrist
            # removes hand/arm translation and leaves finger-specific motion.
            relative_y = y - hand.wrist_y
            state = self._motion.get(finger_id)
            if state is None:
                self._motion[finger_id] = FingerMotion(position=relative_y)
                continue
            smoothed = (
                self._config.smoothing * relative_y
                + (1 - self._config.smoothing) * state.position
            )
            velocity = smoothed - state.position
            state.position = smoothed
            last_hit = self._last_hit.get(finger_id, 0.0)
            available = now - last_hit >= self._config.cooldown_seconds
            if velocity <= -self._config.release_velocity_threshold:
                state.armed = True
            if state.armed and velocity >= self._config.tap_velocity_threshold and available:
                self._last_hit[finger_id] = now
                state.armed = False
                confidence = min(1.0, velocity / (self._config.tap_velocity_threshold * 2))
                predictions.append(TapPrediction(finger_id, confidence))
        return predictions
