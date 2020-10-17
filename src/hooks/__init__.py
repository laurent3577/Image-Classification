from .hooks_core import *
from .swa import SWA
from .kd import KnowledgeDistillation, MEAL_V2
from .vat import VAT


def build_hooks(config):
    hooks = [Validation(), LRCollect(), AccCollect(), LossCollect(), Logging()]
    if config.SWA.USE:
        hooks.append(
            SWA(config.SWA.EPOCH_START, config.SWA.LR, config.SWA.ANNEAL_EPOCH)
        )
    if config.KD.USE:
        hooks.append(KnowledgeDistillation(config.KD.TEACHER_PATH, config.KD.COEFF))
    if config.MEAL.USE:
        hooks.append(MEAL_V2(config.MEAL.TEACHER_PATH, config.MODEL.NUM_CLASSES))
    if config.VAT.USE:
        hooks.append(VAT(eps=config.VAT.EPS, K=config.VAT.K))
    return hooks
