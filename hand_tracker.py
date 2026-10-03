from dataclasses import dataclass

import cv2
import mediapipe as mp

from .classifier import FINGERS
from .config import AppConfig


TIP_IDS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class HandFrame:
    """Normalized fingertip locations from one tracked hand."""

    handedness: str
    wrist: tuple[float, float]
    wrist_y: float
    fingertips: dict[str, tuple[float, float]]
    landmarks: tuple[tuple[float, float], ...]


class HandTracker:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._smoothed: dict[str, list[tuple[float, float]]] = {}
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def detect(self, frame_bgr) -> list[HandFrame]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks or not result.multi_handedness:
            return []
        hands: list[HandFrame] = []
        for landmarks, classification in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):
            # MediaPipe's labels are designed for a mirrored selfie image,
            # which is exactly what the on-screen AirBeat view uses.
            handedness = classification.classification[0].label.lower()
            raw_points = [(point.x, point.y) for point in landmarks.landmark]
            previous = self._smoothed.get(handedness)
            if previous is not None:
                amount = self._config.landmark_smoothing
                points = [
                    (amount * x + (1 - amount) * old_x, amount * y + (1 - amount) * old_y)
                    for (x, y), (old_x, old_y) in zip(raw_points, previous)
                ]
            else:
                points = raw_points
            self._smoothed[handedness] = points
            hands.append(HandFrame(
                handedness=handedness,
                wrist=points[0],
                wrist_y=points[0][1],
                fingertips={
                    finger: points[tip]
                    for finger, tip in zip(FINGERS, TIP_IDS)
                },
                landmarks=tuple(points),
            ))
        return hands

    def close(self) -> None:
        self._hands.close()
