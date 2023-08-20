import importlib
import os
from datetime import datetime
import argparse
import sys

if __name__ == "__main__":
    # ----------- LAUNCH MODULE ---------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name", type=str, required=True, help="module name (without the .py)"
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_name", required=False, default=None)
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    module = importlib.import_module(f"config.{args.file_name}")

    # -------------- setup logging ----------------

    if args.save:
        path_output = "."

        path_output = os.path.join(path_output, "outputs")

        if not os.path.exists(path_output):
            os.mkdir(path_output)
        if args.save_name is not None:
            path_output_file = os.path.join(path_output, args.save_name)
            if os.path.exists(path_output_file):
                path_output_file = os.path.join(
                    path_output,
                    args.save_name + "_" + datetime.now().strftime("%b%d_%H-%M-%S"),
                )
        else:
            path_output_file = os.path.join(
                path_output, datetime.now().strftime("%b%d_%H-%M-%S")
            )
        os.mkdir(path_output_file)

        with open(f"config/{args.file_name}.py", "r") as f:
            save_path = os.path.join(path_output_file, f"{args.file_name}.py")

            with open(save_path, "w") as out:
                for line in f.readlines():
                    print(line, end="", file=out)

        path_command = os.path.join(path_output_file, "command.txt")
        with open(path_command, "w") as f:
            f.write(" ".join(sys.argv))
    else:
        path_output_file = None

    module.launch(path_output_file, args.command)
