import torch


def Accuracy(out, target, ignore_index=None):
    predicted = torch.argmax(out, dim=1)
    if ignore_index is not None:
        re_index = target!=ignore_index
        predicted = predicted[re_index]
        target = target[re_index]
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return correct / total
