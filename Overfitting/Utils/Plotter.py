from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from Utils.Data import DIR, EXP
from Utils.Evaluator import Evaluator


class Plotter:
    accuracy: dict[int, dict[float, list[float]]]
    val_accuracy: dict[int, dict[float, list[float]]]
    file: str

    def __init__(self, evaluator: Optional[Evaluator],
                 file: Optional[int] = None,
                 max_epoch_display: int = -1):
        if file is not None:
            self.file = f'{DIR}/{EXP}{file}'
            self.accuracy, self.val_accuracy, self.config = torch.load(self.file, weights_only=False)
        elif evaluator is not None:
            self.file = ""
            self.accuracy = evaluator.accuracy
            self.val_accuracy = evaluator.val_accuracy
            self.config = evaluator.config
        else:
            raise ValueError("arguments net and file are simultaneously none.")

        self.max_epoch_display = max_epoch_display
        if evaluator is not None:
            self.config.plot_strategy = evaluator.config.plot_strategy

    def build_colors(self):
        self.cols = np.zeros((1, len(self.config.thresholds)))
        for i in range(len(self.config.thresholds)):
            self.cols[0,i]=(0.9*float(i)/len(self.config.thresholds) + 0.05)

    def build_plot(self, idx: int= -1):
        self.build_colors()

        if self.config.plot_strategy == "default":
            if idx < 0:
                return
            dim = self.config.dim[idx]
            self.__default_build_plot(self.accuracy[idx], self.val_accuracy[idx], f"({dim=})")

        elif self.config.plot_strategy == "aggregated":
            accuracy: dict[float, list[float]] = {s: [] for s in self.config.thresholds}
            val_accuracy: dict[float, list[float]] = {s: [] for s in self.config.thresholds}

            for s, dim in product(self.config.thresholds, range(len(self.config.dim))):
                idx = np.argmax(self.accuracy[dim][s])

                accuracy[s].append(self.accuracy[dim][s][idx])
                val_accuracy[s].append(self.val_accuracy[dim][s][idx])

            self.__default_build_plot(accuracy, val_accuracy)

    def __default_build_plot(self,
                             accuracy: dict[float, list[float]],
                             val_accuracy: dict[float, list[float]],
                             title_supp: Optional[str] = None):
        if title_supp is None:
            title_supp = ""

        x_label: str= ""
        if self.config.plot_strategy == "default":
            x_label = "Iterations"
        elif self.config.plot_strategy == "aggregated":
            x_label = "Dimension"


        _, axes = plt.subplots(1,2)
        end = np.inf

        if self.config.plot_strategy == "default":
            end = min(self.max_epoch_display, len(accuracy[self.config.thresholds[0]]))
            if end <= 0:
                end = (len(accuracy[self.config.thresholds[0]]))
        elif self.config.plot_strategy == "aggregated":
            end = max(self.config.dim)

        for i, s in enumerate(self.config.thresholds):
            if s == 0:
                continue

            color = (.5,1-self.cols[0,i],.5)
            label0: str = rf"$\varepsilon=${s:.3f}" if s >= 0.1 else rf"$\varepsilon=${s:2g}"
            label1: Optional[str] = rf"$\varepsilon=${s:.3f}" if (i == 0 or i == len(self.config.thresholds)-1) else None

            if self.config.plot_strategy == "default":
                axes[0].plot(range(len(accuracy[s][:end])), accuracy[s][:end], color = color, label = label0)
                axes[1].plot(range(len(val_accuracy[s][:end])), val_accuracy[s][:end], color = color, label = label1)
            elif self.config.plot_strategy == "aggregated":
                axes[0].plot(self.config.dim, accuracy[s], color = color, label = label0)
                axes[1].plot(self.config.dim, val_accuracy[s], color = color, label = label1)

        axes[0].set_xlim([7, end+200])
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel(x_label)
        axes[0].set_title("training" + title_supp)

        axes[1].set_xlim([7, end+200])
        axes[1].set_xlabel(x_label)
        axes[1].set_title("validation" + title_supp)

        axes[0].legend()


    def plot(self):
        if self.config.print:
            print(f'{self.accuracy=}')
            print(f"{self.val_accuracy=}")

        if self.config.plot_strategy == "default":
            for idx in self.accuracy.keys():
                self.build_plot(idx)
                plt.show()
        elif self.config.plot_strategy == "aggregated":
            self.build_plot()
            plt.show()
