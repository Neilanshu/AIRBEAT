"""Gesture timing and one-control-per-fingertip DJ mappings."""

from dataclasses import dataclass
import time

from .dj_mixer import DjMixer


COMMAND_LABELS = {
    "left:thumb": "BASS", "left:index": "MID/VOC", "left:middle": "HIGHS",
    "left:ring": "REVERB", "left:pinky": "ECHO",
    "right:thumb": "PLAY", "right:index": "VOLUME", "right:middle": "MIX",
    "right:ring": "TRANS", "right:pinky": "CUE",
}


class TapCommandRouter:
    """A tap waits briefly for a possible second tap from the same fingertip."""

    def __init__(self, double_tap_window: float = 0.30) -> None:
        self._window = double_tap_window
        self._pending: dict[str, float] = {}

    def register(self, finger: str, now: float | None = None) -> list[tuple[str, int]]:
        now = time.monotonic() if now is None else now
        first_tap = self._pending.get(finger)
        if first_tap is not None and now - first_tap <= self._window:
            del self._pending[finger]
            return [(finger, 2)]
        self._pending[finger] = now
        return []

    def due(self, now: float | None = None) -> list[tuple[str, int]]:
        now = time.monotonic() if now is None else now
        ready = [finger for finger, started in self._pending.items() if now - started > self._window]
        for finger in ready:
            del self._pending[finger]
        return [(finger, 1) for finger in ready]

    def clear(self) -> None:
        self._pending.clear()


@dataclass
class DjControls:
    """Effect-control state plus the actions that can already affect playback."""

    bass: int = 50
    reverb: int = 0
    vocals: int = 50
    highs: int = 50
    echo: int = 0

    def execute(self, finger: str, taps: int, mixer: DjMixer) -> str:
        direction = 1 if taps == 1 else -1
        if finger == "left:thumb":
            self.bass = self._adjust(self.bass, direction)
            return f"BASS {self.bass}%"
        if finger == "left:index":
            self.vocals = self._adjust(self.vocals, direction)
            return f"MIDS/VOCALS {self.vocals}%"
        if finger == "left:middle":
            self.highs = self._adjust(self.highs, direction)
            return f"HIGHS {self.highs}%"
        if finger == "left:ring":
            self.reverb = self._adjust(self.reverb, direction)
            return f"REVERB {self.reverb}%"
        if finger == "left:pinky":
            self.echo = self._adjust(self.echo, direction)
            return f"ECHO {self.echo}%"

        if finger == "right:thumb":
            mixer.toggle_playing()
            return "PLAY" if mixer.playing else "PAUSE"
        if finger == "right:index":
            mixer.adjust_master_volume(0.10 * direction)
            return f"VOLUME {mixer.master_volume_percent}%"
        if finger == "right:middle":
            mixer.nudge_mix(0.10 * direction)
            return f"MIX B {mixer.mix_percent}%"
        if finger == "right:ring":
            if taps == 1:
                mixer.transition_to_next()
                return "TRANSITION TO B"
            mixer.set_mix(0.0)
            return "RETURN TO A"
        if finger == "right:pinky":
            mixer.cue_next()
            return "CUE B" if taps == 1 else "CUE B RESET"
        return ""

    @staticmethod
    def _adjust(value: int, direction: int) -> int:
        return max(0, min(100, value + direction * 10))

    def effects_label(self) -> str:
        return f"Bass {self.bass}  Reverb {self.reverb}  Vocals {self.vocals}  Highs {self.highs}  Echo {self.echo}"

    def reset_effects(self) -> str:
        """Restore sensible neutral values after experimenting with effects."""
        self.bass, self.vocals, self.highs = 50, 50, 50
        self.reverb, self.echo = 0, 0
        return "EFFECTS RESET"
