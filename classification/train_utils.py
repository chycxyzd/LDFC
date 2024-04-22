import os
import sys
import json
import pickle
import random
import math
import pandas as pd
import seaborn as sn
import numpy as np
import torch.nn as nn

import torch
from tqdm import tqdm
from Kappa import kappa


import matplotlib.pyplot as plt


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    # avg_kappa = 0.0
    all_kappa = 0.0

    y_true_list = []
    y_pred_list = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        pred_classes = pred_classes.cpu()
        labels = labels.cpu()
        y_pred = np.array(pred_classes)
        y_label = np.array(labels)
        y_pred = list(y_pred)
        y_true = list(y_label)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

        # avg_kappa += train_kappa
        # avg_train_acc += accu_num.item() / sample_num

        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}, lr: {:.7f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    # print(all_kappa)
    predd = sum(y_pred_list, [])
    truee = sum(y_true_list, [])
    train_kappa = kappa(predd, truee)

    print("single_epoch_train_kappa = %.4f" % train_kappa)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, train_kappa, truee, predd


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    all_val_kappa = 0.0

    y_true_list = []
    y_pred_list = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        pred_classes = pred_classes.cpu()
        labels = labels.cpu()
        y_pred = np.array(pred_classes)
        y_label = np.array(labels)
        y_pred = list(y_pred)
        y_true = list(y_label)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

        data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
        )

    predd = sum(y_pred_list, [])
    truee = sum(y_true_list, [])
    val_kappa = kappa(predd, truee)
    print("single_val_kappa = %.4f" % val_kappa)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, val_kappa, truee, predd


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        #if "se.fc" in name:
            #print(name)
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or "se.fc" in name:
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())