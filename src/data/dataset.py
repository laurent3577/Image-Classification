from torch.utils.data import Dataset
from torchvision import datasets
from .transforms import build_transforms

class BaseDataset(Dataset):
	def __init__(self, data_dir, split, input_size, transforms=[], target_transform=None):
		self.split = split
		self.input_size = input_size
		self.data = self._get_data(data_dir, self.split)
		self.targets = self._get_targets(data_dir, self.split)
		self.transforms = build_transforms(transforms)
		self.target_transform = target_transform
		self.add_to_sample = None

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = self.transforms(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		sample = {
			"img":img,
			"target":target
		}
		if self.add_to_sample is not None:
			for add_fn in self.add_to_sample:
				sample = add_fn(sample)
		return sample

	def _get_data(self, data_dir, split):
		raise NotImplementedError

	def _get_targets(self, data_dir, split):
		raise NotImplementedError

class CIFAR100(BaseDataset):
	def __init__(self, data_dir, split, input_size, transforms, target_transform):
		super(CIFAR100, self).__init__(data_dir, split, input_size, transforms, target_transform)
		if split == 'val':
			self.val_from_train=True

	def _get_data(self, data_dir, split):
		return datasets.CIFAR100(root=data_dir, train=split!='test', download=True).data
	def _get_targets(self, data_dir, split):
		return datasets.CIFAR100(root=data_dir, train=split!='test', download=True).targets

class CIFAR10(BaseDataset):
	def __init__(self, data_dir, split, input_size, transforms, target_transform):
		super(CIFAR10, self).__init__(data_dir, split, input_size, transforms, target_transform)
		if split == 'val':
			self.val_from_train=True

	def _get_data(self, data_dir, split):
		return datasets.CIFAR10(root=data_dir, train=split!='test', download=True).data
	def _get_targets(self, data_dir, split):
		return datasets.CIFAR10(root=data_dir, train=split!='test', download=True).targets