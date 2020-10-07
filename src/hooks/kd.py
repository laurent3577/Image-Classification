from .hooks_core import Hook
from torchvision import transforms
from ..utils import load_from_path
from ..data import build_transforms
import torch
from tqdm import tqdm
import torch.nn.functional as F

class KnowledgeDistillation(Hook):
	def __init__(self, teacher_path, coeff):
		self.teacher = load_from_path(teacher_path)
		self.teacher.to(self.trainer.device)
		self.coeff = coeff
		self.teacher_targets = {}

	def train_begin(self):
		print("Generating teacher targets...")
		init_transforms = self.trainer.train_loader.dataset.transforms
		self.trainer.train_loader.dataset.transforms=build_transforms(
			[("Resize",{"size":self.trainer.config.DATASET.INPUT_SIZE})],
			self.trainer.train_loader.dataset.normalize_type)
		with torch.no_grad():
			pbar = tqdm(self.trainer.train_loader)
			for sample in pbar:
				sample = self.trainer._to_device(sample)
				pred = self.teacher(sample["img"])
				for i,p in zip(sample["index"], pred):
					self.teacher_targets[int(i.data)] = p
		print("Teacher targets generated")
		self.trainer.train_loader.dataset.add_to_sample.append(self.add_teacher_targets)
		self.trainer.train_loader.dataset.transforms=init_transforms

	def add_teacher_targets(self, sample):
		sample['teacher_targets'] = self.teacher_targets[sample['index']]
		return sample

	def before_backward(self):
		kd_loss = torch.pow(F.softmax(self.trainer.output,1) - F.softmax(self.trainer.input['teacher_targets'],1),2).mean()
		self.trainer.loss += self.coeff * kd_loss
