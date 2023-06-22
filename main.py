import importlib
import os
from datetime import datetime
import argparse

if __name__ == "__main__":
    CONFIG = "manifold-rotations-fs"
    SAVE = True

    module = importlib.import_module(f"config.{CONFIG}")

    # -------------- setup logging ----------------

    if SAVE:
        path_output = "."   
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        path_output = os.path.join(path_output, "outputs")
        if not os.path.exists(path_output):
            os.mkdir(path_output)
        path_output = os.path.join(path_output, current_time)
        os.mkdir(path_output)

        with open(f"config/{CONFIG}.py", "r") as f:
            save_path = os.path.join(path_output, f"{CONFIG}.py")

            with open(save_path, "w") as out:
                for line in f.readlines():
                    print(line, end="", file=out)
    else:
        path_output = None
    # ----------- LAUNCH MODULE ---------------------
    
    module.launch(path_output, argparse.ArgumentParser())
