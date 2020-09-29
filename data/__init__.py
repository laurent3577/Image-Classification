from torchvision import datasets
from .transforms import build_transforms

def build_dataset(config, split, transform=None, target_transform=None):
	if config.DATASET.NAME == "CIFAR100":
		dataset=datasets.CIFAR100(
			root=config.DATASET.ROOT,
			train=split!='test',
			transform=transform,
			target_transform=target_transform,
			download=True)
		if split=="val":
			dataset.val_from_train = True
	elif config.DATASET.NAME == "CIFAR10":
		dataset=datasets.CIFAR10(
			root=config.DATASET.ROOT,
			train=split!='test',
			transform=transform,
			target_transform=target_transform,
			download=True)
		if split=="val":
			dataset.val_from_train = True
	else:
		raise ValueError("Not supported dataset: {}".format(config.DATASET.NAME))
	return dataset