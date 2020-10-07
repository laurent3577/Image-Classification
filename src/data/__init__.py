from .dataset import CIFAR100, CIFAR10

def build_dataset(config, split, transform=None, target_transform=None):
	if config.DATASET.NAME == "CIFAR100":
		CIFAR100(
			data_dir=config.DATASET.ROOT,
			split=split,
			transforms=transform,
			target_transform=target_transform)
	elif config.DATASET.NAME == "CIFAR10":
		CIFAR10(
			data_dir=config.DATASET.ROOT,
			split=split,
			transforms=transform,
			target_transform=target_transform)
	else:
		raise ValueError("Not supported dataset: {}".format(config.DATASET.NAME))
	return dataset