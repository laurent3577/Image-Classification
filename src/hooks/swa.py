from . import Hook
from torch.optim import swa_utils
from copy import deepcopy

class SWA(Hook):
    def __init__(self, epoch_start, swa_lr, anneal_epoch):
        self.epoch_start = epoch_start
        self.swa_lr = swa_lr
        self.anneal_epoch = anneal_epoch

    def train_begin(self):
        self.swa_model = swa_utils.AveragedModel(self.trainer.model)
        swa_epochs = self.trainer.config.OPTIM.EPOCH - self.epoch_start
        anneal_epoch = int(swa_epochs*self.anneal_epoch)
        self.swa_scheduler = swa_utils.SWALR(self.trainer.optim, swa_lr=self.swa_lr, anneal_epochs=anneal_epoch)

    def epoch_end(self):
        if self.trainer.epoch > self.epoch_start:
            self.swa_model.update_parameters(self.trainer.model)
            self.swa_scheduler.step()

    def train_end(self):
        swa_utils.update_bn(self.trainer.train_loader, self.swa_model, self.trainer.device)
        # Copying parameters back to original model
        self.trainer.model = deepcopy(self.swa_model.module)
        self.trainer.val()
