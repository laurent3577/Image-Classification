from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


class ExpAvgMeter:
    def __init__(self, coeff):
        self.coeff = coeff
        self.value = 0
        self.iter = 0

    def update(self, value):
        self.iter += 1
        if self.iter == 1:
            self.value = value
        else:
            self.value = self.coeff * self.value + (1 - self.coeff) * value


class Plotter(object):
    def __init__(self, log_dir="/tmp/"):
        self.viz = SummaryWriter(log_dir)
        self.plots = {}

    def plot(self, plot_title, x, y, legend, xlabel, ylabel):
        self.viz.add_scalars(plot_title, {legend: y}, x)
