from ..hooks_core import Hook
from torchvision import transforms
from ..models import load_from_path
from ..data import build_transforms
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


class KnowledgeDistillation(Hook):
    def __init__(self, teacher_path, coeff):
        self.teachers = [load_from_path(path) for path in teacher_path.split(" ")]
        self.coeff = coeff
        self.teacher_targets = {}

    def train_begin(self):
        print("Generating teacher targets...")
        for teacher in self.teachers:
            teacher.to(self.trainer.device)
            teacher.eval()
        init_transforms = self.trainer.train_loader.dataset.transforms
        self.trainer.train_loader.dataset.transforms = build_transforms(
            [("Resize", {"size": self.trainer.config.DATASET.INPUT_SIZE})],
            self.trainer.train_loader.dataset.normalize_type,
        )
        with torch.no_grad():
            pbar = tqdm(self.trainer.train_loader)
            for sample in pbar:
                sample = self.trainer._to_device(sample)
                pred = torch.mean(
                    torch.stack([teacher(sample["input"]) for teacher in self.teachers]),
                    dim=0,
                )
                for i, p in zip(sample["index"], pred):
                    self.teacher_targets[int(i.data)] = p
        print("Teacher targets generated")
        self.trainer.train_loader.dataset.add_to_sample.append(self.add_teacher_targets)
        self.trainer.train_loader.dataset.transforms = init_transforms

    def add_teacher_targets(self, sample):
        if self.trainer.in_train:
            sample["teacher_targets"] = self.teacher_targets[sample["index"]]
        return sample

    def before_backward(self):
        kd_loss = torch.pow(
            F.softmax(self.trainer.output, 1)
            - F.softmax(self.trainer.batch["teacher_targets"], 1),
            2,
        ).mean()
        self.trainer.loss += self.coeff * kd_loss


class MEAL_V2(KnowledgeDistillation):
    def __init__(self, teacher_path, n_classes, coeff=0):
        super(MEAL_V2, self).__init__(teacher_path, coeff)
        K = 2
        self.discriminator = nn.Sequential(
            nn.Linear(n_classes, n_classes // K),
            nn.ReLU(),
            nn.Linear(n_classes // K, n_classes // K ** 2),
            nn.ReLU(),
            nn.Linear(n_classes // K ** 2, 1),
        )
        self.discr_loss = nn.BCEWithLogitsLoss()

    def train_begin(self):
        new_param_groups = self.trainer.optim.param_groups
        new_param_groups.append(
            {
                "params": self.discriminator.parameters(),
                "lr": self.trainer.config.OPTIM.BASE_LR / 100,
            }
        )
        self.trainer.update_optim(param_groups=new_param_groups)
        self.discriminator.to(self.trainer.device)
        super(MEAL_V2, self).train_begin()

    def get_discr_loss(self):
        x = torch.cat(
            (
                self.discriminator(self.trainer.output),
                self.discriminator(self.trainer.batch["teacher_targets"]),
            )
        )
        y = torch.cat(
            (
                torch.zeros(self.trainer.output.size(0), 1),
                torch.ones(self.trainer.output.size(0), 1),
            )
        ).to(x.device)
        return self.discr_loss(x, y)

    def before_backward(self):
        kld_loss = -torch.sum(
            F.softmax(self.trainer.batch["teacher_targets"], 1)
            * F.log_softmax(self.trainer.output, 1),
            dim=1,
        ).mean()
        discr_loss = self.get_discr_loss()
        if self.coeff:
            self.trainer.loss *= self.coeff
            self.trainer.loss += kld_loss + discr_loss
        else:   # in original paper MEAL doesn't use supervision from hard labels
            self.trainer.loss = kld_loss + discr_loss
