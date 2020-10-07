from .dataset import CIFAR100, CIFAR10
from .transforms import build_transforms

def build_dataset(config, split, transform=None, target_transform=None):
	if config.DATASET.NAME == "CIFAR100":
		return CIFAR100(
			data_dir=config.DATASET.ROOT,
			split=split,
			transforms=transform,
			target_transform=target_transform)
	elif config.DATASET.NAME == "CIFAR10":
		return CIFAR10(
			data_dir=config.DATASET.ROOT,
			split=split,
			transforms=transform,
			target_transform=target_transform)
	else:
		raise ValueError("Not supported dataset: {}".format(config.DATASET.NAME))