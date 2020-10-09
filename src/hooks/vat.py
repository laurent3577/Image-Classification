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

    @staticmethod
    def set_bn_train(m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()

    def _adv_distance(self, pred, x, pert, coeff):
        pred_hat = self.trainer.model(x+coeff*pert)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        return adv_distance

    def batch_begin(self):
        if self.trainer.in_train:
            # VAT loss should be computed before regular forward pass
            x = self.trainer.input["img"]
            with torch.no_grad():
                pred = F.softmax(self.trainer.model(x), dim=1)

            pert = torch.normal(0,1, size=x.shape).to(x.device)
            pert = self._l2_normalize(pert)
            self.trainer.model.apply(self.set_bn_eval)
            for _ in range(self.K):
                pert.requires_grad_()
                adv_distance = self._adv_distance(pred, x, pert, self.xi)
                pert.retain_grad()
                adv_distance.backward()
                pert = self._l2_normalize(pert.grad)
                self.trainer.model.zero_grad()

            self.trainer.model.apply(self.set_bn_train)
            self.lds = self._adv_distance(pred, x, pert, self.eps)

    def before_backward(self):
        self.trainer.loss += self.alpha * self.lds