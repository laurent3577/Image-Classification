from .hooks_core import *
from .swa import SWA
from .kd import KnowledgeDistillation

def build_hooks(config):
	hooks = [Validation(), LRCollect(), AccCollect(), LossCollect()]
	if config.SWA.USE:
		hooks.append(SWA(config.SWA.EPOCH_START, config.SWA.LR, config.SWA.ANNEAL_EPOCH))
	if config.KD.USE:
		hooks.append(KnowledgeDistillation(config.KD.TEACHER_PATH, config.KD.COEFF))
	hooks.append(Logging()) # logging needs to be last on the list
	return hooks