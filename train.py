from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torch import nn, save

from utils.utils import transfer_to


def launch_training(
    training_config,
    training_module: nn.Module,
    dataloader_train,
    path_output: str,
    validation_module: nn.Module = None,
    validation_dataloader=None,
    DEVICE="cuda:0",
):
    optimizer = training_config["optimizer"]
    scheduler = training_config["scheduler"]
    epochs = training_config["epochs"]
    start_epoch = training_config["start_epoch"]
    name_split = training_config["name_split"]

    path_output = os.path.join(path_output, name_split)

    writer = SummaryWriter(log_dir=path_output)

    for epoch in range(start_epoch, epochs):
        print("epoch : ", epoch)
        training_module.train()
        training_module.to(DEVICE)

        for input_data in tqdm(dataloader_train):
            optimizer.zero_grad()

            # ----- INPUT -------
            input_data = transfer_to(input_data, DEVICE)

            # ----- FORWARD ------
            loss = training_module(input_data)
            loss.backward()
            optimizer.step()
            # break

        training_module.eval()
        scheduler.step()

        # ---------- SAVE AND LOG ---------------

        dict_logs = training_module.accumulate_and_get_logs(epoch)

        for name_metric, value in dict_logs.items():
            writer.add_scalar(name_metric, value, epoch)
        
        print(dict_logs)
        
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        if epoch % 10 == 0:
            training_config["start_epoch"] = epoch
            save(
                dict(
                    training_module = training_module,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    epochs = epochs,
                    name_split = name_split,
                    start_epoch = epoch,
                ),
                path_output + "/checkpoint.pt",
            )

        training_module.to("cpu")

        if validation_module is None:
            continue

        # ---------- VALIDATION STEP ---------------------

        validation_module.to(DEVICE)
        validation_module.eval()

        for input_data in tqdm(validation_dataloader):
            # ----- INPUT -------
            input_data = transfer_to(input_data, DEVICE)
            # ----- FORWARD ------
            loss = validation_module(input_data)

        # ---------- SAVE AND LOG ---------------

        dict_logs = validation_module.accumulate_and_get_logs(epoch)
        print("validation : ")
        print(dict_logs)
        validation_module.to("cpu")
        
        for name_metric, value in dict_logs.items():
            writer.add_scalar("validation-" + name_metric, value, epoch)
