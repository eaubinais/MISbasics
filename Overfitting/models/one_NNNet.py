from typing import Any

import torch
from lightning import LightningModule
from Models.RegNet import Config
from sklearn.neighbors import KNeighborsRegressor


class one_NNNet(LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.loss = torch.nn.MSELoss()

        self.classifier = KNeighborsRegressor(n_neighbors=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.Tensor(self.classifier.predict(x))

    def training_step(self, batch: Any) -> None:
        x, y = batch
        self.classifier.fit(x, y)

    def validation_step(self, batch: Any) -> None:
        pass
    def configure_optimizers(self) -> None:
        pass
