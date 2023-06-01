import importlib
import os
from datetime import datetime

if __name__ == "__main__":
    CONFIG = "manifold-rotations"
    module = importlib.import_module(f"train_scripts.{CONFIG}") 

    path_output = "."

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    path_output = os.path.join(path_output, "outputs")
    path_output = os.path.join(path_output, current_time)
    
    os.mkdir(path_output)
    with open(f"train_scripts/{CONFIG}.py", "r") as f:
        save_path = os.path.join(path_output, f"{CONFIG}.py")
        
        with open(save_path, "w") as out: 
            for line in f.readlines():
                print(line, end="", file=out)
    module.launch(path_output)