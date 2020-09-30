from .resnet import *

def build_model(config):
	if config.MODEL.NAME == "resnet18":
		return resnet18(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "resnet34":
		return resnet34(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	elif config.MODEL.NAME == "resnet50":
		return resnet50(config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES)
	else:
		raise ValueError("Not supported model: {}".format(config.MODEL.NAME))