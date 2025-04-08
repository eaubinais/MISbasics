import os

from Utils.Evaluator import Evaluator
from Utils.Plotter import Plotter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()

    plotter = Plotter(evaluator=evaluator,
                      file=evaluator.config.file,
                      max_epoch_display=evaluator.config.max_epoch_display)
    plotter.plot()

