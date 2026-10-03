"""Turn one audio file into ten independent, beat-aligned finger pads."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import soundfile as sf


PAD_IDS = (
    "left:thumb", "left:index", "left:middle", "left:ring", "left:pinky",
    "right:thumb", "right:index", "right:middle", "right:ring", "right:pinky",
)


@dataclass
class SongBeatPads:
    """A tap advances that fingertip through every tenth beat of the song."""

    pads: dict[str, list[pygame.mixer.Sound]]
    positions: dict[str, int]
    beat_count: int

    @classmethod
    def from_file(cls, path: str, sample_rate: int = 44_100) -> "SongBeatPads":
        audio_path = Path(path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Song not found: {audio_path}")

        waveform, source_rate = sf.read(audio_path, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = cls._resample(waveform, source_rate, sample_rate)
        if len(waveform) == 0:
            raise ValueError("The song contains no audio samples.")

        beat_samples = cls._detect_beats(waveform, sample_rate)
        boundaries = cls._boundaries(beat_samples, len(waveform), sample_rate)
        if len(boundaries) < 2:
            raise ValueError("Could not find enough beats in this song.")

        pads = {pad_id: [] for pad_id in PAD_IDS}
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            clip = cls._make_sound(waveform[start:end])
            pads[PAD_IDS[index % len(PAD_IDS)]].append(clip)
        return cls(pads=pads, positions={pad_id: 0 for pad_id in PAD_IDS}, beat_count=len(boundaries) - 1)

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return samples
        source_positions = np.arange(len(samples)) / source_rate
        target_positions = np.arange(int(len(samples) * target_rate / source_rate)) / target_rate
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    @staticmethod
    def _detect_beats(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Find strong energy changes, with a spacing suitable for finger taps."""
        hop = 512
        usable = len(samples) - (len(samples) % hop)
        if usable < hop * 4:
            return np.array([], dtype=int)
        frames = samples[:usable].reshape(-1, hop)
        energy = np.sqrt(np.mean(frames * frames, axis=1))
        onset = np.maximum(np.diff(energy, prepend=energy[0]), 0)
        threshold = np.percentile(onset, 75)
        candidates = [
            index for index in range(1, len(onset) - 1)
            if onset[index] >= threshold and onset[index] >= onset[index - 1] and onset[index] >= onset[index + 1]
        ]
        # Keep at most one onset every 0.22 seconds, choosing the strongest.
        minimum_gap = max(1, int(0.22 * sample_rate / hop))
        chosen: list[int] = []
        for index in sorted(candidates, key=lambda item: onset[item], reverse=True):
            if all(abs(index - accepted) >= minimum_gap for accepted in chosen):
                chosen.append(index)
        return np.array(sorted(index * hop for index in chosen), dtype=int)

    @staticmethod
    def _boundaries(beats: np.ndarray, total_samples: int, sample_rate: int) -> np.ndarray:
        # Beat detection can miss very short or quiet files. In that case, use
        # half-second chunks so the song is still playable as pads.
        valid = beats[(beats > 0) & (beats < total_samples)]
        if len(valid) < 2:
            return np.arange(0, total_samples + 1, sample_rate // 2, dtype=int)
        return np.unique(np.concatenate(([0], valid, [total_samples]))).astype(int)

    @staticmethod
    def _make_sound(clip: np.ndarray) -> pygame.mixer.Sound:
        # A 5 ms fade prevents clicks at slice boundaries.
        samples = clip.astype(np.float32).copy()
        fade = min(220, len(samples) // 2)
        if fade:
            ramp = np.linspace(0, 1, fade, endpoint=False)
            samples[:fade] *= ramp
            samples[-fade:] *= ramp[::-1]
        peak = np.max(np.abs(samples)) if len(samples) else 0
        if peak > 1:
            samples /= peak
        return pygame.sndarray.make_sound((samples * 32767).astype(np.int16))

    def play(self, pad_id: str) -> None:
        clips = self.pads.get(pad_id, [])
        if not clips:
            return
        position = self.positions[pad_id]
        clips[position].play()
        self.positions[pad_id] = (position + 1) % len(clips)

    def reset(self) -> None:
        for pad_id in self.positions:
            self.positions[pad_id] = 0
