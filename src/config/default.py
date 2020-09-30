from yacs.config import CfgNode as CN
import os

_C = CN()
_C.OUTPUT_DIR = os.path.join(os.environ.get("HOME"), "Downloads")
_C.EXP_NAME = ''
_C.RANDOM_SEED = 0
### LOSS FUNCTION

### MODEL
_C.MODEL = CN()
_C.MODEL.NAME = "resnet50"
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.PRETRAINED = True

### OPTIMIZER
_C.OPTIM = CN()
_C.OPTIM.OPTIMIZER = "AdamW"
_C.OPTIM.BASE_LR = 0.001
_C.OPTIM.WEIGHT_DECAY = 0.01
_C.OPTIM.EPOCH=10
_C.OPTIM.BATCH_SIZE = 16
_C.OPTIM.SCHEDULER = CN()
_C.OPTIM.SCHEDULER.TYPE = "Step"
_C.OPTIM.SCHEDULER.STEP_SIZE = 5
_C.OPTIM.SCHEDULER.GAMMA = 0.1
_C.OPTIM.SCHEDULER.COSINE_LR_MIN = 1e-7
_C.OPTIM.SCHEDULER.CYCLE_DIV_FACTOR = 25
### DATASET
_C.DATASET = CN()
_C.DATASET.ROOT = os.path.join(os.environ.get("HOME"), "Downloads")
_C.DATASET.INPUT_SIZE = (32, 32)
_C.DATASET.NAME = "CIFAR100"
_C.DATASET.VAL_SIZE = 0.2
### LOGGING
_C.PLOT_EVERY = 20
_C.VISDOM = True
### DEBUG
_C.DEBUG = CN()
_C.DEBUG.USE = True
_C.DEBUG.DEBUG_EVERY = 100
_C.DEBUG.DETECT_THRESH = 0.3
_C.DEBUG.SAVE_SIZE = 128
### SWA
_C.SWA = CN()
_C.SWA.USE = False
_C.SWA.EPOCH_START = 5
_C.SWA.ANNEAL_EPOCH = 0.5
_C.SWA.LR = 0.00001

def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()