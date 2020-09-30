import os
from ..utils import *

class Hook():
	def train_begin(self):
		return
	def epoch_begin(self):
		return
	def batch_begin(self):
		return
	def before_loss(self):
		return
	def before_backward(self):
		return
	def batch_end(self):
		return
	def epoch_end(self):
		return
	def skip_val(self):
		return
	def train_end(self):
		return
	def val_begin(self):
		return
	def val_end(self):
		return

class Validation(Hook):
	def val_begin(self):
		self.accuracy = 0
		self.loss = 0
		self.total = 0
		self.nb_batch = len(self.trainer.val_loader)

	def batch_end(self):
		if not self.trainer.in_train:
			self.loss += self.trainer.loss
			self.accuracy += acc(self.trainer.output, self.trainer.target)
			self.total += self.trainer.output.size(0)

	def val_end(self):
		print("Validation results: Acc: {0:.2f} ({1}/{2})   Loss: {3:.4f}".format(
			self.accuracy/self.nb_batch*100,
			int(self.accuracy/self.nb_batch*total),
			self.total,
			self.loss/self.nb_batch))

class Logging(Hook):
	def train_begin(self):
		if self.trainer.config.VISDOM:
			self.plotter = Plotter(log_to_filename=os.path.join(self.trainer.config.OUTPUT_DIR, "logs.viz"))
		self.loss_meter = ExpAvgMeter(0.98)
		self.acc_meter = ExpAvgMeter(0.98)

	def batch_end(self):
		if self.trainer.in_train:
			self.loss_meter.update(float(self.trainer.loss.data))
			accuracy = acc(self.trainer.output, self.trainer.target)
			self.acc_meter.update(accuracy*100)
			self.trainer.pbar.set_description(
				'Train Epoch : {0}/{1} Loss : {2:.4f} Acc : {3:.2f} '.format(
					self.trainer.epoch,
					self.trainer.config.OPTIM.EPOCH,
					self.loss_meter.value,
					self.acc_meter.value))
			if self.trainer.config.VISDOM and self.trainer.step % self.trainer.config.PLOT_EVERY == 0:
				self.plotter.plot("Loss", self.trainer.step, self.loss_meter.value, "Loss", "Step", "Value")
				lr = self.trainer.optim.param_groups[0]['lr']
				self.plotter.plot("LR", self.trainer.step, lr, "LR", "Step", "Value")

