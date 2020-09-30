from . import Hook
from torch.optim import swa_utils
from copy import deepcopy

class SWA(Hook):
	def __init__(self, epoch_start, swa_lr):
		self.epoch_start = epoch_start
		self.swa_lr = swa_lr

	def train_begin(self):
		self.swa_model = swa_utils.AveragedModel(self.trainer.model)
		self.swa_scheduler = swa_utils.SWALR(self.trainer.optim, swa_lr=self.swa_lr)

	def epoch_end(self):
		if self.trainer.epoch > self.epoch_start:
          self.swa_model.update_parameters(self.trainer.model)
          self.swa_scheduler.step()

    def train_end(self):
    	swa_utils.update_bn(self.trainer.train_loader, self.swa_model)
    	# Copying parameters back to original model
    	self.trainer.model = deepcopy(self.swa_model.module)
