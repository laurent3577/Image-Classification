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
	def stop_epoch(self):
		return
	def epoch_end(self):
		return
	def skip_val(self):
		return
	def stop_train(self):
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
			int(self.accuracy/self.nb_batch*self.total),
			self.total,
			self.loss/self.nb_batch))
		self.trainer.to_plot.append(["Loss", self.trainer.step, self.loss/self.nb_batch, "Val Loss", "Step", "Value"])
		self.trainer.to_plot.append(["Acc", self.trainer.step, self.accuracy/self.nb_batch*100, "Val Acc", "Step", "Value"])

class Logging(Hook):
	def train_begin(self):
		if self.trainer.config.PLOT:
			self.plotter = Plotter(log_dir=os.path.join(self.trainer.config.OUTPUT_DIR, "logs"), visdom=self.trainer.config.VISDOM)
			self.trainer.to_plot = []

	def batch_end(self):
		if self.trainer.in_train:
			self.trainer.pbar.set_description(
				'Train Epoch : {0}/{1} Loss : {2:.4f} Acc : {3:.2f} '.format(
					self.trainer.epoch,
					self.trainer.config.OPTIM.EPOCH,
					self.trainer.state['Loss_last'],
					self.trainer.state['Acc_last']))
			if self.trainer.config.PLOT and self.trainer.step % self.trainer.config.PLOT_EVERY == 0:
				self.trainer.to_plot.append(["Loss", self.trainer.step, self.trainer.state['Loss_last'], "Train Loss", "Step", "Value"])
				self.trainer.to_plot.append(["Acc", self.trainer.step, self.trainer.state['Acc_last'], "Train Acc", "Step", "Value"])
				self.trainer.to_plot.append(["LR", self.trainer.step, self.trainer.state['LR_last'], "LR", "Step", "Value"])
				self._plot()

	def _plot(self):
		for plot in self.trainer.to_plot:
			self.plotter.plot(*plot)
		self.trainer.to_plot = []

	def val_end(self):
		self._plot()

class EarlyStop(Hook):
	"""
	Allows to stop training after certain number of iterations or epoch.
	Number of iterations / epoch processed will be exactly stop_iter or
	stop_epoch.
	"""
	def __init__(self, stop_iter=None, stop_epoch=None):
		assert (stop_iter is not None) or (stop_epoch is not None)
		self.stop_iter = stop_iter
		self.stop_epoch = stop_epoch
	def stop_epoch(self):
		if self.stop_iter is not None:
			return self.trainer.step >= self.stop_iter
	def stop_train(self):
		if self.stop_epoch is not None:
			return self.trainer.epoch > self.stop_epoch
		else:
			return self.trainer.step >= self.stop_iter

class Collect(Hook):
	def __init__(self, collect_type):
		self.collect_type = collect_type

	def train_begin(self):
		if getattr(self.trainer, 'state', False):
			self.trainer.state = {}

	def batch_end(self):
		self._update(*self._collect())

	def _update(self, k, v):
		if collect_type == 'list':
			if k in self.trainer.state:
				self.trainer.state[k].append(v)
			else:
				self.trainer.state[k] = [v]
		elif collect_type == 'last':
			self.trainer.state[k] = v

	def _collect(self):
		return None, None

class LRCollect(Collect):
	def _collect(self):
		lr = self.trainer.optim.param_groups[0]['lr']
		return 'LR_{}'.format(self.collect_type), lr

class LossCollect(Collect):
	def __init__(self, collect_type='last'):
		super(LossCollect, self).__init__(collect_type)
		self.meter = ExpAvgMeter(0.98)
	def _collect(self):
		self.meter.update(float(self.trainer.loss.data))
		return 'Loss_{}'.format(self.collect_type), self.meter.value

class AccCollect(Collect):
	def __init__(self, collect_type='last'):
		super(AccCollect, self).__init__(collect_type)
		self.meter = ExpAvgMeter(0.98)
	def _collect(self):
		accuracy = acc(self.trainer.output, self.trainer.target)
		self.meter.update(accuracy*100)
		return 'Acc_{}'.format(self.collect_type), self.meter.value
