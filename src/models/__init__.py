from .resnet import *
from .densenet import *

def build_model(config):
	if config.MODEL.NAME == "resnet18":
		return resnet18(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "resnet34":
		return resnet34(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "resnet50":
		return resnet50(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "resnet101":
		return resnet101(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "resnext50_32x4d":
		return resnext50_32x4d(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "resnext101_32x8d":
		return resnext101_32x8d(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "densenet121":
		return densenet121(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "densenet161":
		return densenet161(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "densenet169":
		return densenet169(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "densenet201":
		return densenet201(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	else:
		raise ValueError("Not supported model: {}".format(config.MODEL.NAME))