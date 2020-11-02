import torch


def acc(out, target):
    predicted = torch.argmax(out, dim=1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return correct / total
