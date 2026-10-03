"""Model boundary for current and future fingertip tap classifiers."""

from dataclasses import dataclass
from typing import Protocol, Sequence


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class TapPrediction:
    """A classifier's decision for one camera frame."""

    finger: str
    confidence: float


class TapClassifier(Protocol):
    """Replace this with a custom CNN without changing the app pipeline.

    `frames` is a chronological RGB frame sequence.  The current baseline also
    receives landmark data via its own update method, because it does not use
    pixels. A CNN implementation should expose this method and emit only new
    tap events (not a continuous 'down' state).
    """

    def predict(self, frames: Sequence[object]) -> Sequence[TapPrediction]:
        ...


class CnnTapClassifier:
    """Reserved integration point for a trained temporal CNN.

    Add model loading, frame preprocessing, and inference here once training
    data exists. It remains intentionally uninstantiated in the baseline.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        raise NotImplementedError(
            "Train/export a tap model before enabling CnnTapClassifier. "
            "The webcam prototype uses LandmarkTapDetector meanwhile."
        )
