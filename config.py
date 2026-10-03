from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    # A 960×540 input is responsive on most laptops while keeping fingertips
    # reliable enough for gesture control.
    frame_width: int = 960
    frame_height: int = 540
    max_hands: int = 2
    # Landmark y-coordinates increase downward. This is the per-frame downward
    # velocity required for a hit after smoothing.
    tap_velocity_threshold: float = 0.006
    # A finger must lift by at least this amount before it can trigger again.
    release_velocity_threshold: float = 0.003
    smoothing: float = 0.45
    cooldown_seconds: float = 0.16
    min_detection_confidence: float = 0.75
    min_tracking_confidence: float = 0.75
    model_complexity: int = 1
    landmark_smoothing: float = 0.65
    gesture_cooldown_seconds: float = 0.8
    long_pinch_seconds: float = 0.55
    # Put an MP3, WAV, or OGG file at this path to play it as a backing track.
    # Leave as None to use AirBeat as a standalone instrument.
    backing_track_path: str | None = str(Path(__file__).parent / "assets" / "song.mp3")
    playlist_paths: tuple[str, ...] = (
        str(Path(__file__).parent / "assets" / "song.mp3"),
        str(Path(__file__).parent / "assets" / "song_2.mp3"),
        str(Path(__file__).parent / "assets" / "song_3.mp3"),
        str(Path(__file__).parent / "assets" / "song_4.mp3"),
    )
    cnn_model_path: str = str(Path(__file__).parent / "models" / "tap_cnn.pt")
    cnn_window_frames: int = 12
    cnn_confidence_threshold: float = 0.80
    backing_track_volume: float = 0.55


CONFIG = AppConfig()
