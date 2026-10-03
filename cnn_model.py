"""Small temporal CNN for recognizing taps from a sequence of fingertip motion."""

import torch
from torch import nn


class TemporalTapCNN(nn.Module):
    """Classifies five fingers from 12 frames of wrist-relative tip movement."""

    def __init__(self, fingers: int = 5) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(fingers, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.20), nn.Linear(64, fingers),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence shape: batch × 5 fingers × 12 chronological frames
        return self.network(sequence)
