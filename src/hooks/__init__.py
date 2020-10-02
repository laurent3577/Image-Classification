from .hooks_core import *
from .swa import SWA

def build_hooks(config):
	hooks = [Validation(), LRCollect('last'), AccCollect('last'), LossCollect('last')]
	if config.SWA.USE:
		hooks.append(SWA(config.SWA.EPOCH_START, config.SWA.LR, config.SWA.ANNEAL_EPOCH))
	hooks.append(Logging()) # logging needs to be last on the list
	return hooks