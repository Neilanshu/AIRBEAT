"""Playlist-based two-deck playback and reliable crossfades for AirBeat."""

from pathlib import Path
import time

import numpy as np
import pygame
import soundfile as sf


class DjMixer:
    """One audible deck plus a cued next song, cycling through a playlist."""

    def __init__(self, sample_rate: int = 44_100) -> None:
        self._sample_rate = sample_rate
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
        pygame.mixer.set_num_channels(16)
        self._deck_a = pygame.mixer.Channel(12)
        self._deck_b = pygame.mixer.Channel(13)
        self._playlist: list[Path] = []
        self._sounds: dict[Path, pygame.mixer.Sound] = {}
        self._current_index = 0
        self._mix = 0.0
        self._master_volume = 0.80
        self._playing = False
        self._transition_started: float | None = None
        self._transition_duration = 2.0

    def load_playlist(self, paths: tuple[str, ...]) -> bool:
        self._playlist = [Path(path) for path in paths if path and Path(path).is_file()]
        if not self._playlist:
            return False
        self._current_index = 0
        self._playing = True
        self._deck_a.play(self._sound(self._playlist[0]), loops=-1)
        self._cue_following_track()
        self._apply_mix()
        return True

    def _sound(self, path: Path) -> pygame.mixer.Sound:
        if path not in self._sounds:
            samples, source_rate = sf.read(path, dtype="float32")
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            if source_rate != self._sample_rate:
                source_x = np.arange(len(samples)) / source_rate
                target_x = np.arange(int(len(samples) * self._sample_rate / source_rate)) / self._sample_rate
                samples = np.interp(target_x, source_x, samples).astype(np.float32)
            self._sounds[path] = pygame.sndarray.make_sound(
                np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            )
        return self._sounds[path]

    def _cue_following_track(self) -> None:
        if not self.has_next_track:
            self._deck_b.stop()
            return
        self._deck_b.play(self._sound(self._playlist[self.next_index]), loops=-1)
        if not self._playing:
            self._deck_b.pause()

    def _apply_mix(self) -> None:
        self._deck_a.set_volume(float(np.cos(self._mix * np.pi / 2)) * self._master_volume)
        self._deck_b.set_volume(
            float(np.sin(self._mix * np.pi / 2)) * self._master_volume if self.has_next_track else 0
        )

    @property
    def has_next_track(self) -> bool:
        return len(self._playlist) > 1

    @property
    def next_index(self) -> int:
        return (self._current_index + 1) % len(self._playlist)

    @property
    def current_track_name(self) -> str:
        return self._playlist[self._current_index].stem if self._playlist else "No song"

    @property
    def next_track_name(self) -> str:
        return self._playlist[self.next_index].stem if self.has_next_track else "—"

    @property
    def track_count(self) -> int:
        return len(self._playlist)

    @property
    def mix_percent(self) -> int:
        return round(self._mix * 100)

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def master_volume_percent(self) -> int:
        return round(self._master_volume * 100)

    def toggle_playing(self) -> None:
        if not self._playlist:
            return
        if self._playing:
            self._deck_a.pause()
            self._deck_b.pause()
        else:
            self._deck_a.unpause()
            self._deck_b.unpause()
        self._playing = not self._playing

    def nudge_mix(self, amount: float) -> None:
        if not self.has_next_track:
            return
        self._transition_started = None
        self._mix = float(np.clip(self._mix + amount, 0, 1))
        self._apply_mix()

    def set_mix(self, amount: float) -> None:
        self._transition_started = None
        self._mix = float(np.clip(amount, 0, 1))
        self._apply_mix()

    def adjust_master_volume(self, amount: float) -> None:
        self._master_volume = float(np.clip(self._master_volume + amount, 0, 1))
        self._apply_mix()

    def cue_next(self) -> None:
        """Restart the next track silently, ready for a transition."""
        if self.has_next_track:
            self._cue_following_track()
            self._apply_mix()

    def transition_to_next(self) -> None:
        if self.has_next_track and self._playing:
            self._transition_started = time.monotonic()

    def tick(self) -> None:
        if self._transition_started is None:
            return
        progress = (time.monotonic() - self._transition_started) / self._transition_duration
        self._mix = min(1.0, progress)
        self._apply_mix()
        if progress >= 1:
            self._transition_started = None
            self._deck_a.stop()
            self._deck_a, self._deck_b = self._deck_b, self._deck_a
            self._current_index = self.next_index
            self._mix = 0.0
            self._cue_following_track()
            self._apply_mix()
