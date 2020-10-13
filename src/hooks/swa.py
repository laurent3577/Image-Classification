from . import Hook
from torch.optim import swa_utils
from copy import deepcopy
import torch


class SWA(Hook):
    def __init__(self, epoch_start, swa_lr, anneal_epoch):
        self.epoch_start = epoch_start
        self.swa_lr = swa_lr
        self.anneal_epoch = anneal_epoch

    def train_begin(self):
        self.swa_model = swa_utils.AveragedModel(self.trainer.model)
        swa_epochs = self.trainer.config.OPTIM.EPOCH - self.epoch_start
        anneal_epoch = int(swa_epochs * self.anneal_epoch)
        self.swa_scheduler = swa_utils.SWALR(
            self.trainer.optim, swa_lr=self.swa_lr, anneal_epochs=anneal_epoch
        )

    def epoch_end(self):
        if self.trainer.epoch > self.epoch_start:
            self.trainer.scheduler.update_on_step = False
            self.swa_model.update_parameters(self.trainer.model)
            self.swa_scheduler.step()

    def update_bn(self):
        momenta = {}
        for module in self.swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = self.swa_model.training
        self.swa_model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        for input in self.trainer.train_loader:
            input = input["img"]
            input = input.to(self.trainer.device)
            self.swa_model(input)

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        self.swa_model.train(was_training)

    def train_end(self):
        self.update_bn()
        # Copying parameters back to original model
        self.trainer.model = deepcopy(self.swa_model.module)
        print("Validation results with SWA weights")
        self.trainer.val()
