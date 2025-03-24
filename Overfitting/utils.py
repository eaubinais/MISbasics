import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataloader
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar
from nn import RegNet, RegNetConfig
from torch.utils.data import Dataset

DIR = r"./overfitting/exp"
EXP = "exp_"

class MISRatioModule:
    def __init__(
        self,
        model: Optional[LightningModule],
        thresholds: list[float]
    ):
        assert(model is not None)
        self.module = model
        self.args = None
        self.thresholds = thresholds
        self.accuracy: dict[float, list[float]] = {}
        self.val_accuracy: dict[float, list[float]] = {}

        for s in self.thresholds:
            self.accuracy[s] = []
            self.val_accuracy[s] = []

        self.unreduced_loss = self.__unreduced_loss_function(self.module)

        self.module.training_step = self.__wrapped_step(
            self.module, self.module.training_step, validation=False
        )
        self.module.validation_step = self.__wrapped_step(
            self.module, self.module.validation_step, validation=True
        )

    def __unreduced_loss_function(self,
                                  instance: LightningModule):
        assert(hasattr(instance, "loss"))
        return getattr(instance, "loss").__class__(reduction = 'none')

    def __wrapped_step(
        self,
        instance: LightningModule,
        func_to_wrap: Callable[..., Any],
        validation: bool = False
    ) -> Callable[..., Any]:

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

            return out

        return new_func

    def ratio_of_losses_under_threshold(self, y_pred: torch.Tensor, y: torch.Tensor, threshold: float) -> float:
        loss = self.unreduced_loss(y_pred, y)
        return len(np.where(loss <= threshold)[0]) / float(y_pred.shape[0])


    def compute_thresholds(self, y_pred: torch.Tensor, y: torch.Tensor) -> dict[float, float]:
        result : dict[float, float] = {
            s : self.ratio_of_losses_under_threshold(y_pred, y, s)
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

@dataclass
class EvaluatorConfig:
    dim: int = 10
    num_of_data: int = 1000
    epochs: int = 3000
    lr: float = 0.1
    net_size: int = 4096
    max_epoch_display: int = -1
    file: Optional[int] = None
    thresholds: list[float] = field(default_factory=lambda: [0.1])

    def to_RegNetConfig(self) -> RegNetConfig:
        return RegNetConfig(self.dim, self.net_size, self.lr, self.thresholds)

    def from_dict(self, d: argparse.Namespace):
        for key, val in d._get_kwargs():
            if hasattr(self, key):
                setattr(self, key, val)

class Evaluator:
    net: MISRatioModule

    def __init__(self):
        args = self.get_args()
        self.config = EvaluatorConfig()
        self.config.from_dict(args)

    def evaluate(self) -> None:
        if self.config.file is not None:
            return

        dataset = SyntheticDataset(self.config.dim, self.config.num_of_data, lambda x : np.sin(x))
        val_dataset = SyntheticDataset(self.config.dim, 1000, lambda x : np.sin(x))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.num_of_data,shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1000,shuffle=False)

        net = RegNet(self.config.to_RegNetConfig())
        self.net = MISRatioModule(net, self.config.thresholds)

        callbacks = [
            RichProgressBar(refresh_rate=5)
        ]

        trainer = Trainer(max_epochs=self.config.epochs,
                          check_val_every_n_epoch=1,
                          num_sanity_val_steps=0,
                          enable_checkpointing=False,
                          callbacks=callbacks,
                          )

        trainer.fit(self.net.module, dataloader, val_dataloader)

        if not os.path.exists("./overfitting/exp"):
            os.mkdir("./overfitting/exp")
        if not os.path.isdir("./overfitting/exp"):
            raise FileNotFoundError("'./overfitting/exp' is not a directory.")
        if len(os.listdir(DIR))==0:
            val = 0
        else:
            val = max(int(re.search(fr'(?<={EXP}).*', f)[0])  for f in os.listdir(DIR))

        torch.save((self.net.accuracy, self.net.val_accuracy, self.config), f"{DIR}/{EXP}{val+1}")

    def get_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        parser.add_argument("--dim", type=int, default=10)
        parser.add_argument("--num_of_data", type=int, default=1000)
        parser.add_argument("--epochs", type=int, default=3000)
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--net_size", type=int, default=4096)
        parser.add_argument("--thresholds", type=float, nargs="+", default=[0.000001, 0.00001, 0.0001, 0.001,0.01,0.1,0.5])
        parser.add_argument("--max_epoch_display", type=int, default=-1)
        parser.add_argument("--file", type=int, default=None)

        return parser.parse_args()

class Plotter:
    accuracy: dict[float, list[float]]
    val_accuracy: dict[float, list[float]]
    file: str

    def __init__(self, evaluator: Optional[Evaluator],
                 file: Optional[int] = None,
                 validation_included: bool = True,
                 max_epoch_display: int = -1):
        if file is not None:
            self.file = f'{DIR}/{EXP}{file}'
            self.accuracy, self.val_accuracy, self.config = torch.load(self.file)
        elif evaluator is not None:
            self.file = ""
            self.accuracy = evaluator.net.accuracy
            self.val_accuracy = evaluator.net.val_accuracy
            self.config = evaluator.config
        else:
            raise ValueError("arguments net and file are simultaneously none.")

        self.validation_included = validation_included
        self.max_epoch_display = max_epoch_display

    def build_colors(self):
        self.cols = np.zeros((1, len(self.config.thresholds)))
        for i in range(len(self.config.thresholds)):
            self.cols[0,i]=(0.9*float(i)/len(self.config.thresholds) + 0.05)

    def build_plot(self):
        self.build_colors()

        _, axes = plt.subplots(1,2)
        end = min(self.max_epoch_display, len(self.accuracy[self.config.thresholds[0]]))
        if end <= 0:
            end = (len(self.accuracy[self.config.thresholds[0]]))

        for i, s in enumerate(self.config.thresholds):
            if s == 0:
                continue

            color = (.5,1-self.cols[0,i],.5)
            label0: str = rf"$\varepsilon=${s:.3f}" if s >= 0.1 else rf"$\varepsilon=${s:2g}"
            axes[0].plot(range(len(self.accuracy[s][:end])), self.accuracy[s][:end], color = color, label = label0)

            label1: Optional[str] = rf"$\varepsilon=${s:.3f}" if (i == 0 or i == len(self.config.thresholds)-1) else None
            axes[1].plot(range(len(self.val_accuracy[s][:end])), self.val_accuracy[s][:end], color = color, label = label1)

        axes[0].set_xlim([7, end+200])
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel("Iterations")
        axes[0].set_title("training")

        axes[1].set_xlim([7, end+200])
        axes[1].set_xlabel("Iterations")
        axes[1].set_title("validation")

        axes[0].legend()

    def plot(self):
        self.build_plot()
        plt.show()
