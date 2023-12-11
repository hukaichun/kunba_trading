import os
import datetime
import typing as tp

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
from tqdm import tqdm


import dataset as DS
import model as md

def load_config(config_path:str):
    with open(config_path, 'r', encoding="utf8") as f:
        config = yaml.safe_load(f)
    return config

def print_config(the_dict:tp.Dict, prefix=""):
    for k, v in the_dict.items():
        if isinstance(v, dict):
            print(prefix, k, ":")
            print_config(v, prefix+'\t')
        else:
            print(prefix, k, ":", v)

def get_save_to_path(prefix:str):
    current_datetime = datetime.datetime.now()
    current_time_string = current_datetime.strftime("%Y%m%d_%H%M%S")
    full_path_name = "/".join(["./saved_models", prefix, current_time_string])

    if not os.path.isdir(full_path_name):
        os.makedirs(full_path_name)

    return full_path_name


def training_loop(*training_set_loaders: torch.utils.data.DataLoader,
                  model: torch.nn.Module,
                  criterion: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  gradient_clip:float,
                  DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    losses_log = {}
    training_set_loaders = [training_set_loader for training_set_loader in training_set_loaders]
    np.random.shuffle(training_set_loaders)
    model.train()
    for training_set_loader in training_set_loaders:
        model = model.to(DEVICE)
        pbar = tqdm(training_set_loader)
        pbar.set_description("training phase")
        for images, labels in pbar:
            images = images.to(torch.float32).to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(images)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            losses_log.setdefault("losses", []).append(loss.detach().cpu().numpy())
    return losses_log


def validation_loop(*validation_set_loaders: torch.utils.data.DataLoader,
                  model: torch.nn.Module,
                  criterion: torch.nn.Module,
                  DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    losses_log = {}
    model.eval()
    m = torch.nn.Softmax(1)
    with torch.no_grad():
        for validation_set_loader in validation_set_loaders:
            model = model.to(DEVICE)
            pbar = tqdm(validation_set_loader)
            pbar.set_description("validation phase")
            for images, labels in pbar:
                images = images.to(torch.float32).to(DEVICE)
                labels = labels.to(DEVICE)
                log_pred = model(images)
                pred = torch.argmax(log_pred, 1)
                acc = torch.sum(pred==labels)/len(labels)
                losses_log.setdefault("acc", []).append(acc.detach().cpu().numpy())
        return losses_log





def main(config_path:str="./configs/config.yaml"):
    config = load_config(config_path)
    print_config(config)

    model = md.SimpleCNN(2)
    
    training_config = config["training_config"]
    optimizer = torch.optim.Adadelta(model.parameters(), lr = training_config["lr"], rho=training_config["rho"], eps=training_config["eps"])
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 
                                                                         factor=training_config["learning_rate_reduce_factor"], 
                                                                         patience=training_config["patience"],
                                                                         min_lr=training_config["minimum_learning_rate"])
    training_data_config = config["training_data"]
    training_data = DS.TimeSeqSigmentSign(
        data=training_data_config["data_path"],
        feature_columns=training_data_config["feature_columns"],
        label_column=training_data_config["label_column"],
        window_width=training_data_config["window_width"]
    )
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=training_data_config["batch_size"],
                                                                      shuffle=training_data_config["shuffle"],
                                                                      num_workers=training_data_config["num_workers"],
                                                                      prefetch_factor=training_data_config["prefetch_factor"])

    validation_data_config = config["validation_data"]
    validation_data = DS.TimeSeqSigmentSign(
        data=validation_data_config["data_path"],
        feature_columns=validation_data_config["feature_columns"],
        label_column=validation_data_config["label_column"],
        window_width=validation_data_config["window_width"]
    )
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_data_config["batch_size"],
                                                                      shuffle=validation_data_config["shuffle"],
                                                                      num_workers=validation_data_config["num_workers"],
                                                                      prefetch_factor=validation_data_config["prefetch_factor"])



    save_path = get_save_to_path(config["experiment_name"])
    writer = SummaryWriter(save_path)
    print(save_path)

    best_loss = {}
    
    training_epoch = training_config["training_episode"]
    for epoch in range(training_epoch):
        training_log = training_loop(training_data_loader, model=model, criterion=criterion, optimizer=optimizer, gradient_clip=training_config["grad_clip"])
        total_loss = sum(training_log["losses"])
        learning_rate_scheduler.step(total_loss, epoch)
        for k,v in training_log.items():
            writer.add_histogram(k, np.asarray(v), epoch)
        writer.add_scalar("lr", learning_rate_scheduler._last_lr[-1], epoch)

        if best_loss.setdefault("best_training_loss", np.inf)>total_loss:
            best_loss["best_training_loss"] = total_loss
            torch.save(model.state_dict(), f"{save_path}/best_train.pth")

        validation_log = validation_loop(validation_data_loader, model=model, criterion=criterion)
        total_val_acc = sum(validation_log["acc"])
        for k,v in validation_log.items():
            writer.add_histogram(f"validation_{k}", np.asarray(v), epoch)
            writer.add_scalar(f"validation_{k}_avg", np.asarray(v).mean(), epoch)

        if best_loss.setdefault("best_val_loss", -1)<total_val_acc:
            best_loss["best_val_loss"] = total_val_acc
            torch.save(model.state_dict(), f"{save_path}/best_val.pth")
        
        if epoch%training_config["save_freq"]==0:
            torch.save(model.state_dict(), f"{save_path}/epoch_{epoch+1}.pth")
        
        print(epoch)

if __name__ == "__main__":
    main()