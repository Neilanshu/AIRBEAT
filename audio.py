"""Small generated drum kit: keeps the starter project asset-free."""

import numpy as np
import pygame
from pathlib import Path


class DrumKit:
    def __init__(self, sample_rate: int = 44_100) -> None:
        pygame.mixer.pre_init(sample_rate, -16, 1, 512)
        pygame.init()
        self._sample_rate = sample_rate
        self._sounds = {
            "left:thumb": self._sound(self._kick()),
            "left:index": self._sound(self._snare()),
            "left:middle": self._sound(self._hat()),
            "left:ring": self._sound(self._clap()),
            "left:pinky": self._sound(self._tom()),
            "right:thumb": self._sound(self._note(261.63)),  # C4
            "right:index": self._sound(self._note(293.66)),  # D4
            "right:middle": self._sound(self._note(329.63)), # E4
            "right:ring": self._sound(self._note(392.00)),   # G4
            "right:pinky": self._sound(self._note(523.25)),  # C5
        }

    def _sound(self, waveform: np.ndarray) -> pygame.mixer.Sound:
        samples = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
        return pygame.sndarray.make_sound(samples)

    def _envelope(self, length: float, decay: float) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, length, int(self._sample_rate * length), endpoint=False)
        return t, np.exp(-t * decay)

    def _kick(self) -> np.ndarray:
        t, env = self._envelope(0.32, 13)
        frequency = 150 * np.exp(-t * 14) + 42
        return 0.9 * np.sin(2 * np.pi * np.cumsum(frequency) / self._sample_rate) * env

    def _snare(self) -> np.ndarray:
        t, env = self._envelope(0.19, 19)
        return (0.72 * np.random.uniform(-1, 1, len(t)) + 0.28 * np.sin(2 * np.pi * 185 * t)) * env

    def _hat(self) -> np.ndarray:
        t, env = self._envelope(0.075, 55)
        return np.random.uniform(-1, 1, len(t)) * env * 0.35

    def _clap(self) -> np.ndarray:
        t, env = self._envelope(0.16, 18)
        pulse = ((np.sin(2 * np.pi * 27 * t) > 0).astype(float) * 0.6 + 0.4)
        return np.random.uniform(-1, 1, len(t)) * env * pulse * 0.45

    def _tom(self) -> np.ndarray:
        t, env = self._envelope(0.25, 11)
        return np.sin(2 * np.pi * 118 * t) * env * 0.75

    def _note(self, frequency: float) -> np.ndarray:
        t, env = self._envelope(0.42, 6.5)
        tone = np.sin(2 * np.pi * frequency * t)
        harmonic = 0.25 * np.sin(2 * np.pi * frequency * 2 * t)
        return (tone + harmonic) * env * 0.45

    def play(self, finger: str) -> None:
        self._sounds[finger].play()

    def load_backing_track(self, path: str | None, volume: float) -> bool:
        """Load and immediately loop a local song. Returns False when absent."""
        if not path or not Path(path).is_file():
            return False
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(loops=-1)
        return True

    def toggle_backing_track(self) -> bool:
        """Pause/resume the track; returns whether it is now playing."""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            return False
        pygame.mixer.music.unpause()
        return True

    def close(self) -> None:
        pygame.quit()
