
import os
from typing import Any
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from MMML.utils import transfer_to, TrainingConfig
from MMML.callbacks import SaveStateCallback



def classical_training(writer, training_config : TrainingConfig, DEVICE):
    optimizer = training_config.optimizer
    scheduler = training_config.scheduler
    epochs = training_config.epochs
    start_epoch = training_config.start_epoch
    training_module = training_config.training_module
    dataloader_train = training_config.dataloader
    callbacks = training_config.callbacks

    for epoch in range(start_epoch, epochs):
        print("epoch : ", epoch)
        training_module.train()
        training_module.to(DEVICE)

        for input_data in tqdm(dataloader_train):
            optimizer.zero_grad()
            input_data = transfer_to(input_data, DEVICE)
            loss = training_module(input_data)
            loss.backward()
            optimizer.step()

            
        scheduler.step()
        training_module.eval()
        
        # ---------- SAVE AND LOG ---------------
        training_config.start_epoch = training_config.start_epoch +1 

        training_module.to("cpu")
        for callback in callbacks:
            callback(writer, training_config)
        
def multiple_gradient_training(writer, training_config : TrainingConfig, DEVICE):
    optimizer = training_config.optimizer
    scheduler = training_config.scheduler
    epochs = training_config.epochs
    start_epoch = training_config.start_epoch
    training_modules = training_config.training_module
    dataloader_train = training_config.dataloader
    callbacks = training_config.callbacks

    for epoch in range(start_epoch, epochs):
        print("epoch : ", epoch)

        for training_module in training_modules:
            training_module.to(DEVICE)
        for input_data in tqdm(dataloader_train):
            for training_module in training_modules:
                optimizer.zero_grad()
                training_module.train()
                optimizer.step()    
                input_data = transfer_to(input_data, DEVICE)
                loss = training_module(input_data)
                loss.backward()

        scheduler.step()     
        for training_module in training_modules:
            training_module.to("cpu")
            training_module.eval()
        

        # ---------- SAVE AND LOG ---------------
        training_config.start_epoch= training_config.start_epoch +1 

        for callback in callbacks:
            callback(writer, training_config)

TRAINING_NAME_TO_FUNCTION = {
    "classical_training" : classical_training,
    "multiple_gradient_training" : multiple_gradient_training
}

def launch_training(
    training_configs : dict[str, TrainingConfig],
    path_output: str = None,
    save_each_n_epoch = 10,
    DEVICE="cuda:0",
):
    if path_output is not None:
        print("output file : ", path_output)
    
    saving_callback = SaveStateCallback(path_output, training_configs, save_each_n_epoch)
    
    for name_split, training_config in training_configs.items():
        
        training_config.name_split = name_split 

        training_config.callbacks = training_config.callbacks + [saving_callback]
        training_methode = training_config.training_methode
        if path_output is not None :
            path_output = os.path.join(path_output, name_split)
            writer = SummaryWriter(log_dir=path_output)
        else:
            writer = None

        TRAINING_NAME_TO_FUNCTION[training_methode](writer, training_config, DEVICE)

    saving_callback(training_config)