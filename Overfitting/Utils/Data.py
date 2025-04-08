import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from lightning import LightningModule
from Models.LinNet import LinNet
from Models.one_NNNet import one_NNNet
from Models.RegNet import RegNet
from torch.utils.data import Dataset

DIR = r"./overfitting/exp"
EXP = "exp_"

@dataclass
class element:
    cls: type
    func: Callable[[Any], Any]

__elements__ = {
    "RegNet" : element(RegNet, lambda x: np.sin(x)).__dict__,
    "LinNet" : element(LinNet, lambda x: x).__dict__,
    "KNN" : element(one_NNNet, lambda x: np.sin(x)).__dict__
}

class MISRatioModule:
    def __init__(
        self,
        model: Optional[LightningModule],
        thresholds: list[float]
    ):
        assert(model is not None)
        self.module = model
        self.thresholds = thresholds
        self.accuracy: dict[float, list[float]] = {s: [] for s in self.thresholds}
        self.val_accuracy: dict[float, list[float]] = {s: [] for s in self.thresholds}

        self.unreduced_loss = self.__unreduced_loss_function(self.module)

        self.module.training_step = self.__wrapped_step(
            self.module, self.module.training_step, validation=False
        )
        self.module.validation_step = self.__wrapped_step(
            self.module, self.module.validation_step, validation=True
        )

    def __unreduced_loss_function(self, instance: LightningModule):
        assert(hasattr(instance, "loss"))
        return getattr(instance, "loss").__class__(reduction = 'none')

    def __wrapped_step(
        self,
        instance: LightningModule,
        func_to_wrap: Callable[..., Any],
        validation: bool = False
    ) -> Callable[..., Any]:

        @functools.wraps(func_to_wrap)
        def new_func(batch: Any):
            out = func_to_wrap(batch)

            with torch.no_grad():
                x, y = batch
                y_pred = instance(x)
                checks = self.compute_thresholds(y_pred, y)

                if validation:
                    for s, v in checks.items():
                        self.val_accuracy[s].append(v)
                else:
                    for s, v in checks.items():
                        self.accuracy[s].append(v)
                    instance.log("reached_threshold", self.accuracy[min(self.thresholds)][-1])

            return out

        return new_func

    def compute_thresholds(self, y_pred: torch.Tensor, y: torch.Tensor) -> dict[float, float]:
        result : dict[float, float] = {
            s : len(np.where(self.unreduced_loss(y_pred, y) <= s)[0]) / float(y_pred.shape[0])
            for s in self.thresholds
        }

        return result

class SyntheticDataset(Dataset[Any]):
    def __init__(self, dim: int, num_data: int, f: Callable[..., Any]):
        self.dim = dim
        self.num_data = num_data
        self.func = f
        self.X, self.y = self.generate_data()

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def generate_data(self):
        data = [np.random.normal(0,1,self.dim) for _ in range(self.num_data)]
        data = np.array([x / np.linalg.norm(x) for x in data])

        noise = [np.random.normal(0,.1,1) for _ in range(self.num_data)]
        vec = np.array([int(j==0) for j in range(self.dim)])
        y = np.array([self.func(data[i]@vec * np.pi) + noise[i] for i in range(self.num_data)])

        data = torch.Tensor(data).float()
        y = torch.Tensor(y).float()

        return data, y
