import argparse
from tqdm import tqdm
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from optim import build_opt
from models import build_model
from config import config, update_config
from data import build_dataset, build_transforms
from utils import ExpAvgMeter, Plotter
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train  AnoVAEGAN')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def acc(out, target):
    _, predicted = torch.max(out, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return correct/total

def val(model, val_loader, loss_fn):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    pbar = tqdm(val_loader)
    accuracy = 0
    loss = 0
    total = len(val_loader)
    for img, target in pbar:
        img = img.to(device)
        target = target.to(device)
        out = model(img)
        loss += loss_fn(out,target)
        accuracy += acc(out, target)
        pbar.set_description('Validation Acc : {0:.2f}'.format(accuracy*100))
    print("Validation results: Acc: {0:.2f} ({1}/{2})   Loss: {3:.4f}".format(accuracy/total*100, int(total*accuracy), total, loss/total))


def main():
    args = parse_args()
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    print(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(config)
    model.to(device)

    transforms = build_transforms([
        ("Resize", {"size": config.DATASET.INPUT_SIZE}),
        ("HorizontalFlip", None),
        ("VerticalFlip", None)
        ], config)
    val_transforms = build_transforms([
        ("Resize", {"size": config.DATASET.INPUT_SIZE})
        ], config)
    dataset = build_dataset(config, split='train', transform=transforms)
    val_dataset = build_dataset(config, split='val', transform=val_transforms)
    if getattr(val_dataset, "val_from_train", False):
        num_train = len(dataset)
        indices = list(range(num_train))
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices)
        split = int(np.floor(config.DATASET.VAL_SIZE * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        val_dataset.data = val_dataset.data[val_idx]
        dataset.data = dataset.data[train_idx]

    train_loader =DataLoader(dataset, batch_size=config.OPTIM.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.OPTIM.BATCH_SIZE, shuffle=False)

    model_opt, model_scheduler = build_opt(config, model, len(train_loader))

    loss_fn = nn.CrossEntropyLoss()

    loss_meter = ExpAvgMeter(0.98)
    acc_meter = ExpAvgMeter(0.98)
    if config.VISDOM:
        plotter = Plotter(log_to_filename=os.path.join(config.OUTPUT_DIR, "logs.viz"))

    step = 0
    for e in range(config.OPTIM.EPOCH):
        model.train()
        pbar = tqdm(train_loader)
        for img, target in pbar:
            step += 1
            img = img.to(device)
            target = target.to(device)
            out = model(img)
            loss =  loss_fn(out,target)

            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
            if model_scheduler.update_on_step:
                model_scheduler.step()

            loss_meter.update(float(loss.data))
            accuracy = acc(out, target)
            acc_meter.update(accuracy*100)

            pbar.set_description('Train Epoch : {0}/{1} Loss : {2:.4f} Acc : {3:.2f} '.format(e+1, config.OPTIM.EPOCH, loss_meter.value, acc_meter.value))

            if config.VISDOM and step%config.PLOT_EVERY == 0:
                plotter.plot("Loss", step, loss_meter.value, "Loss", "Step", "Value")
                model_lr = model_opt.param_groups[0]['lr']
                plotter.plot("LR", step, model_lr, "Model LR", "Step", "Value")
        if not model_scheduler.update_on_step:
            model_scheduler.step()
        val(model, val_loader, loss_fn)
        save_path = os.path.join(config.OUTPUT_DIR, config.EXP_NAME + "_checkpoint.pth")
        torch.save({
            'cfg':config,
            'params':model.state_dict()}, save_path)


if __name__ == '__main__':
    main()