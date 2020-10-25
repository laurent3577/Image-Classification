import numpy as np
from .dataset import CIFAR100, CIFAR10
from .transforms import build_transforms
from torch.utils.data.sampler import (
    SubsetRandomSampler,
    SequentialSampler,
    RandomSampler,
)


def build_dataset(config, split, transform=None, target_transform=None):
    if config.DATASET.NAME == "CIFAR100":
        return CIFAR100(
            data_dir=config.DATASET.ROOT,
            split=split,
            transforms=transform,
            target_transform=target_transform,
        )
    elif config.DATASET.NAME == "CIFAR10":
        return CIFAR10(
            data_dir=config.DATASET.ROOT,
            split=split,
            transforms=transform,
            target_transform=target_transform,
        )
    else:
        raise ValueError("Not supported dataset: {}".format(config.DATASET.NAME))

def build_samplers(train_dataset, val_dataset, config):
    if getattr(val_dataset, "val_from_train", False):
        num_train = len(train_dataset)
        indices = list(range(num_train))
        if config.RANDOM_SEED:
            np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices)
        split = int(np.floor(config.DATASET.VAL_SIZE * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
    else:
        train_sampler = RandomSampler(dataset)
        val_sampler = SequentialSampler(val_dataset)

    return train_sampler, val_sampler
