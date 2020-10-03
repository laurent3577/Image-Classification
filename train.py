import argparse
from tqdm import tqdm
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler
from src import *
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train  Classification Model')

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


def main():
    args = parse_args()
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    print(config)
    model = build_model(config)

    transforms = build_transforms([
        ("RandomResizedCrop", {"size": config.DATASET.INPUT_SIZE, "scale": (0.5,1.0)}),
        ("Perspective", None),
        ("HorizontalFlip", None),
        ("VerticalFlip", None)
        ], config)
    val_transforms = build_transforms([
        ("Resize", {"size": config.DATASET.INPUT_SIZE})
        ], config)
    dataset = build_dataset(config, split='train', transform=transforms)
    val_dataset = build_dataset(config, split='val', transform=val_transforms)
    train_sampler = RandomSampler(dataset)
    val_sampler = SequentialSampler(val_dataset)
    if getattr(val_dataset, "val_from_train", False):
        num_train = len(dataset)
        indices = list(range(num_train))
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices)
        split = int(np.floor(config.DATASET.VAL_SIZE * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)


    train_loader = DataLoader(dataset, batch_size=config.OPTIM.BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.OPTIM.BATCH_SIZE, sampler=val_sampler)

    opt, scheduler = build_opt(
        optimizer_name=config.OPTIM.OPTIMIZER,
        base_lr=config.OPTIM.BASE_LR,
        weight_decay=config.OPTIM.WEIGHT_DECAY,
        scheduler_name=config.OPTIM.SCHEDULER.TYPE,
        step_size=config.OPTIM.SCHEDULER.STEP_SIZE,
        gamma=config.OPTIM.SCHEDULER.GAMMA,
        cosine_lr_min=config.OPTIM.SCHEDULER.COSINE_LR_MIN,
        cycle_div_factor=config.OPTIM.SCHEDULER.CYCLE_DIV_FACTOR,
        epochs=config.OPTIM.EPOCH,
        steps_per_epoch=len(train_loader),
        model=model)

    loss_fn = nn.CrossEntropyLoss()
    hooks = build_hooks(config)
    trainer = Trainer(model, train_loader, val_loader, opt, scheduler, loss_fn, hooks, config)

    if config.LR_FINDER.USE:
        trainer.lr_finder(min_lr=config.LR_FINDER.MIN_LR, max_lr=config.LR_FINDER.MAX_LR)
    else:
        trainer.train(epoch=config.OPTIM.EPOCH)


if __name__ == '__main__':
    main()