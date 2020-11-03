import torch


def to_one_hot(y, ndim):
    y_onehot = torch.zeros(y.size(0), ndim).to(y.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot

def FocalLoss(gamma, ignore_index=None, eps=1e-8):
    def _FocalLoss(x,y):
        y = to_one_hot(y, x.size(1))
        logpt = torch.log_softmax(x+eps,1)*y + torch.log(1-torch.softmax(x,1)+eps)*(1-y)
        pt = torch.softmax(x,1)*y + (1-torch.softmax(x,1))*(1-y)
        fl = -((1-pt)**gamma)*logpt
        if ignore_index is not None:
        	fl[:,ignore_index] = 0
        return fl.sum()/fl.size(0)
    return _FocalLoss