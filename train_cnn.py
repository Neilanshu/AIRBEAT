"""Train AirBeat's temporal CNN from labeled .npz examples."""

from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .cnn_model import TemporalTapCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tap_examples.npz")
    parser.add_argument("--output", default="models/tap_cnn.pt")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    data = np.load(args.data)
    x = torch.tensor(data["features"], dtype=torch.float32)
    y = torch.tensor(data["labels"], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model, loss_fn = TemporalTapCNN(), nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, args.output)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()
