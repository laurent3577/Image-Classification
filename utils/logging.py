from visdom import Visdom
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
    """Plots to Visdom"""
    def __init__(self, env_name='main', log_to_filename='/tmp/logs.viz'):
        self.viz = Visdom(port=VISDOM_PORT, log_to_filename=log_to_filename)
        self.env = env_name
        self.plots = {}

    def plot(self, plot_title, x, y, legend, xlabel, ylabel):
        if plot_title not in self.plots:
            self.plots[plot_title] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[legend],
                title=plot_title,
                xlabel=xlabel,
                ylabel=ylabel
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[plot_title], name=legend, update='append')