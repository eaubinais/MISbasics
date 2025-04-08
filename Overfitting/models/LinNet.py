from typing import Any

import torch
from lightning import LightningModule
from Models.RegNet import Config


class LinNet(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.linear1 = torch.nn.Linear(config.dim, 1, bias=False)

        self.loss = torch.nn.MSELoss()

        self.losses : list[torch.Tensor] = []
        self.test_losses : list[torch.Tensor] = []

        torch.nn.init.xavier_normal_(self.linear1.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        return out

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
