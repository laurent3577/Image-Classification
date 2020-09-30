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
        ("Resize", {"size": config.DATASET.INPUT_SIZE}),
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

    opt, scheduler = build_opt(config, model, len(train_loader))

    loss_fn = nn.CrossEntropyLoss()
    hooks = build_hooks(config)
    trainer = Trainer(model, train_loader, val_loader, opt, scheduler, loss_fn, hooks, config)

    trainer.train(epoch=config.OPTIM.EPOCH)


if __name__ == '__main__':
    main()