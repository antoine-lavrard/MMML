import importlib
import os
from datetime import datetime
from torch import load

from MMML.train import launch_training

if __name__ == "__main__":
    PATH_RESUME = "outputs//resnet_lookahead_Aug07_21-07-18"

    path_state = os.path.join(PATH_RESUME, "checkpoint.pt")
    train_configs = load(path_state)
    launch_training(train_configs, PATH_RESUME)
