import contextlib
from .hooks_core import Hook
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            # m.track_running_stats ^= True
            m.eval() # as suggested https://github.com/lyakaap/VAT-pytorch/issues/12
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


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

    def _adv_distance(self, pred, x, pert):
        pred_hat = self.trainer.model(x+pert)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        return adv_distance

    def batch_begin(self):
        # VAT loss should be computed before regular forward pass
        x = self.trainer.input["img"]
        with torch.no_grad():
            pred = F.softmax(self.trainer.model(x), dim=1)

        pert = torch.rand(x.shape).sub(0.5).to(x.device)
        pert = self._l2_normalize(pert)

        with _disable_tracking_bn_stats(self.trainer.model):
            for _ in range(self.K):
                pert.requires_grad_()
                pert = self.xi * pert
                adv_distance = self._adv_distance(pred, x, pert)
                adv_distance.backward()
                print(pert.grad)
                pert = self._l2_normalize(pert.grad)
                self.trainer.model.zero_grad()

            pert = self.eps * pert
            self.lds = self._adv_distance(pred, x, pert)

    def before_backward(self):
        self.trainer.loss += self.alpha * self.lds