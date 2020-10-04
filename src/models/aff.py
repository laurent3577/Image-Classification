import torch.nn as nn
import torch


def conv1x1block(in_channels, out_channels, use_bn, use_act):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
    if use_bn:
        modules.append(nn.BatchNorm2d(num_features=out_channels))
    if use_act:
        modules.append(nn.LeakyReLU())
    return nn.Sequential(*modules)

class MSCam(nn.Module):
	def __init__(self, in_channels, ratio):
		super(MSCam, self).__init__()
		self._global = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			conv1x1block(in_channels, int(in_channels/ratio), True, True),
			conv1x1block(int(in_channels/ratio), in_channels, True, False))
		self._local = nn.Sequential(
			conv1x1block(in_channels, int(in_channels/ratio), True, True),
			conv1x1block(int(in_channels/ratio), in_channels, True, False))

	def forward(self, x):
		out = x * torch.sigmoid(self._local(x) + self._global(x))
		return out
		
class AFF(nn.Module):
	def __init__(self, in_channels, ratio=16):
		super(AFF, self).__init__()
		self.mscam = MSCam(in_channels, ratio)

	def forward(self, identity, resid):
		x = identity + resid
		x = self.mscam(x)*identity + (1-self.mscam(x))*resid
		return x