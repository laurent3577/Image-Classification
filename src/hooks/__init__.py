from .hooks_core import *
from .swa import SWA

def build_hooks(config):
	hooks = [Logging(), Validation()]
	if config.SWA.USE:
		hooks.append(SWA(config.SWA.EPOCH_START, config.SWA.LR, config.SWA.ANNEAL_EPOCH))
	return hooks