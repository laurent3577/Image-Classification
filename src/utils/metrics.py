import torch

def acc(out, target):
    _, predicted = torch.max(out, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return correct/total