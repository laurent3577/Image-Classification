from .hooks_core import Hook
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAT(Hook):
    """
    Implements Virtual Adversarial Training regularization introduced in
    https://arxiv.org/pdf/1704.03976.pdf
    """
    def __init__(self, alpha=1, xi=1e-6, eps=2.0, K=1):
        self.xi = xi
        self.eps = eps
        self.K = K
        self.alpha = alpha

    @staticmethod
    def _l2_normalize(d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    @staticmethod
    def set_bn_eval(m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.track_running_stats = False

    @staticmethod
    def set_bn_train(m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
            # m.track_running_stats = True

    def _adv_distance(self, target, p_logit):
        logp_hat = F.log_softmax(p_logit, dim=1)
        adv_distance = F.kl_div(logp_hat, target, reduction='batchmean')
        return adv_distance

    def before_backward(self):
        self.trainer.model.apply(self.set_bn_eval) # disable batch stats update
        x = self.trainer.input["img"]
        with torch.no_grad():
            pred = F.softmax(self.trainer.model(x), dim=1).detach()

        pert = torch.randn_like(x)
        pert = self._l2_normalize(pert)
        for _ in range(self.K):
            x_pert = x.data + self.xi * pert
            x_pert.requires_grad_()
            p_d_logit = self.trainer.model(x_pert)
            adv_distance = self._adv_distance(pred, p_d_logit)
            x_pert.retain_grad()
            adv_distance.backward()
            pert = self._l2_normalize(x_pert.grad)
            self.trainer.model.zero_grad()

        x_adv = x + self.eps * pert
        p_adv_logit = self.trainer.model(x_adv)
        lds = self._adv_distance(pred, p_adv_logit)
        self.trainer.loss += self.alpha * lds
        self.trainer.model.apply(self.set_bn_train)