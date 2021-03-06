from tqdm import tqdm
from torch import optim
from .hooks_core import EarlyStop, LRCollect, LossCollect
from .optim import build_opt
import torch
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self, model, train_loader, val_loader, optim, scheduler, loss_fn, hooks, config, device=None, metrics={}
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self._register_hooks(hooks)
        self.metrics = metrics
        self.step = 0
        self.config = config
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.batch_error_warn = False
        self.save_dir = os.path.join(config.OUTPUT_DIR, f"{datetime.datetime.now():%Y%m%d%H%M}")
        os.makedirs(self.save_dir, exist_ok=True)

    def _register_hooks(self, hooks):
        self.hooks = hooks
        for hk in self.hooks:
            hk.trainer = self

    def _add_hook(self, hook, index=None):
        hook.trainer = self
        if index is None:
            index = len(self.hooks) + 1
        self.hooks.insert(index, hook)

    def _add_hooks(self, hooks, indexes=None):
        if indexes is None:
            indexes = [None for i in range(len(hooks))]
        assert len(hooks) == len(indexes)
        for hook, index in zip(hooks, indexes):
            self._add_hook(hook, index)

    def _hook(self, method):
        out = False
        self.hooks.sort(key=lambda hk: getattr(getattr(hk, method),"priority",1),reverse=True)
        for hk in self.hooks:
            try:
                out = out or getattr(hk, method)()
            except:
                print("Failed to apply {} on {}".format(method, hk))
                raise
        return out

    def _to_device(self, x):
        if isinstance(x, list):
            return [self._to_device(e) for e in x]
        elif isinstance(x, tuple):
            return tuple([self._to_device(e) for e in x])
        elif isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        else:
            try:
                return x.to(self.device)
            except Exception as e:
                if not self.batch_error_warn:
                    self.batch_error_warn = True
                    print(f"EXCEPTION ON SETTING BATCH TO DEVICE : {e}")
                return x

    def _process_epoch(self):
        if self.in_train:
            self.model.train()
            self.pbar = tqdm(self.train_loader)
        else:
            self.model.eval()
            self.pbar = tqdm(self.val_loader)
        for batch in self.pbar:
            if self.in_train:
                self.step += 1
            self.batch = batch
            self.model_kwargs = {}
            self._hook("batch_begin")
            self.batch = self._to_device(self.batch)
            self.model_kwargs = self._to_device(self.model_kwargs)
            self.target = self.batch["target"]
            self._process_batch()
            self._hook("batch_end")
            if self._hook("stop_epoch"):
                return

    def _process_batch(self):
        self.optim.zero_grad()
        self.output = self.model(*self.batch["input"], **self.model_kwargs)
        self._hook("before_loss")
        self.loss = self.loss_fn(self.output, self.target)
        if self.in_train:
            self._hook("before_backward")
            self.loss.backward()
            self.optim.step()
            if self.scheduler.update_on_step:
                self.scheduler.step()

    def train(self, epoch):
        self.epoch = 1
        self._hook("train_begin")
        for i in range(epoch):
            self.in_train = True
            self._hook("epoch_begin")
            self._process_epoch()
            self._hook("epoch_end")
            if not self._hook("skip_val"):
                self.val()
            if not self.scheduler.update_on_step:
                self.scheduler.step()
            self.epoch += 1
            if self._hook("stop_train"):
                self._hook("train_end")
                self.save_ckpt("final.pth")
                return
        self._hook("train_end")
        self.save_ckpt("final.pth")

    def val(self):
        self.in_train = False
        self._hook("val_begin")
        with torch.no_grad():
            self._process_epoch()
        self._hook("val_end")

    def lr_finder(self, min_lr=1e-7, max_lr=10, nb_iter=500):
        self.update_optim(
            base_lr=min_lr,
            scheduler_name="Exp",
            gamma=float(np.exp(np.log(max_lr / min_lr) / nb_iter)),
        )
        self._add_hooks(
            [EarlyStop(iter_stop=nb_iter), LRCollect("list"), LossCollect("list")]
        )
        self.train(epoch=nb_iter)
        lrs = self.state["LR_list"]
        loss = self.state["Loss_list"]
        plt.plot(lrs, loss)
        plt.xscale("log")
        plt.show()

    def save_ckpt(self, name=None):
        if name is None:
            save_path = os.path.join(
                self.save_dir,
                "_".join([self.config.EXP_NAME, "checkpoint.pth"]),
            )
        else:
            save_path = os.path.join(
                self.save_dir, "_".join([self.config.EXP_NAME, name])
            )
        torch.save({"cfg": self.config, "params": self.model.state_dict()}, save_path)

    def update_optim(self, **kwargs):
        """
        Base_lr affects basemodel parameters and all param groups for which lr was not set.
        """
        build_opt_params = {
            "param_groups": [
                {"params": pg["params"], "lr": pg["lr"]}
                for pg in self.optim.param_groups
            ],
            "optimizer_name": self.config.OPTIM.OPTIMIZER,
            "base_lr": self.config.OPTIM.BASE_LR,
            "weight_decay": self.config.OPTIM.WEIGHT_DECAY,
            "scheduler_name": self.config.OPTIM.SCHEDULER.TYPE,
            "step_size": self.config.OPTIM.SCHEDULER.STEP_SIZE,
            "gamma": self.config.OPTIM.SCHEDULER.GAMMA,
            "cosine_lr_min": self.config.OPTIM.SCHEDULER.COSINE_LR_MIN,
            "cycle_div_factor": self.config.OPTIM.SCHEDULER.CYCLE_DIV_FACTOR,
            "epochs": self.config.OPTIM.EPOCH,
            "steps_per_epoch": len(self.train_loader),
        }
        config_map = {
            "optimizer_name": "OPTIM.OPTIMIZER",
            "base_lr": "OPTIM.BASE_LR",
            "weight_decay": "OPTIM.WEIGHT_DECAY",
            "scheduler_name": "OPTIM.SCHEDULER.TYPE",
            "step_size": "OPTIM.SCHEDULER.STEP_SIZE",
            "gamma": "OPTIM.SCHEDULER.GAMMA",
            "cosine_lr_min": "OPTIM.SCHEDULER.COSINE_LR_MIN",
            "cycle_div_factor": "OPTIM.SCHEDULER.CYCLE_DIV_FACTOR",
            "epochs": "OPTIM.EPOCH",
        }
        update_config = []
        for k, v in kwargs.items():
            build_opt_params[k] = v
            if k in config_map:
                update_config += [config_map[k], v]

        self.config.defrost()
        self.config.merge_from_list(update_config)
        self.config.freeze()
        if "base_lr" in kwargs:
            build_opt_params["param_groups"][0]["lr"] = build_opt_params["base_lr"]

        self.optim, self.scheduler = build_opt(**build_opt_params)
