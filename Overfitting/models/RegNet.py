from __future__ import annotations  # type:ignore

from dataclasses import dataclass, field
from typing import Any

import torch
from lightning import LightningModule


@dataclass
class Config:
    dim: int
    size: int
    lr: float
    threshold: list[float] = field(default_factory=lambda: [0.1])

class RegNet(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.linear1 = torch.nn.Linear(config.dim, config.size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(config.size, 1)

        self.loss = torch.nn.MSELoss()

        self.losses : list[torch.Tensor] = []
        self.test_losses : list[torch.Tensor] = []

        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def training_step(self, batch: Any) -> torch.Tensor:
        loss = self.step(batch)
        self.losses.append(loss)
        return loss

    def validation_step(self, batch: Any) -> torch.Tensor:
        loss = self.step(batch)
        self.test_losses.append(loss)
        return loss

    def step(self, batch: Any) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        return loss
