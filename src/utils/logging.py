from visdom import Visdom
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

VISDOM_PORT = os.environ.get('VISDOM_PORT', default=8097)

class ExpAvgMeter():
    def __init__(self, coeff):
        self.coeff = coeff
        self.mv_avg = 0
        self.value = 0
        self.iter = 0

    def update(self, value):
        self.iter += 1
        if self.iter==1:
            self.mv_avg = value
            self.value = value
        else:
            self.mv_avg = self.coeff * self.mv_avg + (1-self.coeff) * value
            self.value = self.mv_avg / (1-self.coeff**self.iter)

class Plotter(object):
    def __init__(self, log_dir='/tmp/', visdom=False):
        self.visdom = visdom
        if self.visdom:
            self.viz = Visdom(port=VISDOM_PORT, log_to_filename=os.path.join(log_dir, 'logs.viz'))
            self.env = 'main'
        else:
            self.viz = SummaryWriter(log_dir)
        self.plots = {}

    def plot(self, plot_title, x, y, legend, xlabel, ylabel):
        if self.visdom:
            if plot_title not in self.plots:
                self.plots[plot_title] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                    legend=[legend],
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel
                ))
            else:
                self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[plot_title], name=legend, update='append')
        else:
            self.viz.add_scalars(plot_title, {legend: y}, x)