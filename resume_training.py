import importlib
import os
from datetime import datetime
from torch import load

from MMML.train import launch_training

if __name__ == "__main__":
    PATH_RESUME = "outputs//resnet_mixup_rotation_Aug28_22-29-08"

    path_state = os.path.join(PATH_RESUME, "checkpoint.pt")
    train_configs = load(path_state)
    # TODO : fix it (save all configs, why ono )
    launch_training([train_configs], PATH_RESUME)
