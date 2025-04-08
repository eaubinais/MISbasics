import argparse
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.utilities import rank_zero_only
from Models.RegNet import Config
from tqdm.auto import tqdm
from Utils.Data import DIR, EXP, MISRatioModule, SyntheticDataset, __elements__

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

@dataclass
class EvaluatorConfig:
    dim: list[int] = field(default_factory=lambda: [10])
    num_of_data: list[int] = field(default_factory=lambda: [1000])
    validation_data: int = 1000
    epochs: int = 3000
    lr: float = 0.1
    net_size: int = 4096
    max_epoch_display: int = -1
    file: Optional[int] = None
    thresholds: list[float] = field(default_factory=lambda: [0.1])
    print: bool = False
    model: str = "RegLin"
    steps: int = 0
    plot_strategy: str = "default"
    warnings: bool = False

    def to_Config(self, idx: int) -> Config:
        return Config(self.dim[idx], self.net_size, self.lr, self.thresholds)

    def from_dict(self, d: argparse.Namespace):
        fdim_found: bool = False

        for key, val in d._get_kwargs():
            if key == "fdim" and val is not None:
                fdim_found = True
                m = re.match(r"(\d+)-(\d+)-(\d+)", val)
                assert(m is not None)
                setattr(self, "dim", range(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            elif key == "dim":
                if not fdim_found:
                    setattr(self, key, val)
            elif hasattr(self, key):
                setattr(self, key, val)

        if len(self.num_of_data) == 1 and len(self.dim) > 1:
            self.num_of_data = [self.num_of_data[0]]*len(self.dim)
        elif len(self.num_of_data) > 1 and len(self.dim) == 1:
            self.dim = [self.dim[0]]*len(self.num_of_data)
        assert(len(self.dim)==len(self.num_of_data))
        self.steps = len(self.dim)

class Evaluator:
    net: MISRatioModule

    def __init__(self):
        self.config = EvaluatorConfig()
        self.config.from_dict(self.get_args())
        self.accuracy: dict[int, Any] = {}
        self.val_accuracy: dict[int, Any] = {}

    @rank_zero_only
    def evaluate(self) -> None:
        if self.config.file is not None:
            return
        func: Callable[[Any], Any] = __elements__[self.config.model]["func"]
        cls_: type = __elements__[self.config.model]["cls"]

        for step in tqdm(range(self.config.steps), leave=False, desc="Number of step"):

            dataset = SyntheticDataset(self.config.dim[step], self.config.num_of_data[step], func)
            val_dataset = SyntheticDataset(self.config.dim[step], self.config.validation_data, func)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.num_of_data[step],shuffle=False)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.validation_data,shuffle=False)

            net = cls_(self.config.to_Config(step))
            self.net = MISRatioModule(net, self.config.thresholds)

            callbacks: list[Any] = [
                LearningRateMonitor(logging_interval=None),
                RichProgressBar(refresh_rate=5,leave=True),
                EarlyStopping(monitor= "reached_threshold",
                              stopping_threshold=.99,
                              check_on_train_epoch_end=True,
                              mode="max",
                              min_delta=-1
                              )
            ]

            if self.config.epochs <= 0:
                self.config.epochs = 1

            trainer = Trainer(max_epochs=self.config.epochs,
                            check_val_every_n_epoch=1,
                            num_sanity_val_steps=0,
                            enable_checkpointing=False,
                            callbacks=callbacks,
                            accelerator="cpu",
                            reload_dataloaders_every_n_epochs=self.config.epochs+1,
                            enable_model_summary=False,
                            )
            if self.config.warnings:
                trainer.fit(self.net.module, dataloader, val_dataloader)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore", category=UserWarning)
                    trainer.fit(self.net.module, dataloader, val_dataloader)


            self.accuracy[step] = self.net.accuracy
            self.val_accuracy[step] = self.net.val_accuracy

            if step < self.config.steps -1:
                del net
                del self.net
                del trainer


        if not os.path.exists("./overfitting/exp"):
            os.mkdir("./overfitting/exp")
        if not os.path.isdir("./overfitting/exp"):
            raise FileNotFoundError("'./overfitting/exp' is not a directory.")
        if len(os.listdir(DIR))==0:
            val = 0
        else:
            val = max(int(re.search(fr'(?<={EXP}).*', f)[0])  for f in os.listdir(DIR))

        torch.save((self.accuracy, self.val_accuracy, self.config), f"{DIR}/{EXP}{val+1}")

    def get_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        parser.add_argument("--dim", type=int, nargs="+", default=[10])
        parser.add_argument("--fdim", type=str, required=False)
        parser.add_argument("--num_of_data", type=int, nargs="+", default=[1000])
        parser.add_argument("--validation_data", type=int, default= 1000)
        parser.add_argument("--epochs", type=int, default=3000)
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--net_size", type=int, default=4096)
        parser.add_argument("--thresholds", type=float, nargs="+", default=[0.000001, 0.00001, 0.0001, 0.001,0.01,0.1,0.5])
        parser.add_argument("--max_epoch_display", type=int, default=-1)
        parser.add_argument("--file", type=int, default=None)
        parser.add_argument("--print", action="store_true", default=False)
        parser.add_argument("--model", type=str, default="RegNet")
        parser.add_argument("--plot_strategy", type=str, default="default")
        parser.add_argument("--warnings", action="store_true", default=False)

        return parser.parse_args()
