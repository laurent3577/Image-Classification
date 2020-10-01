from efficientnet_pytorch import EfficientNet

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
			'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

def _EfficientNet(arch, pretrained, **kwargs):
	if pretrained:
		model = EfficientNet.from_pretrained(arch, **kwargs)
	else:
		model = EfficientNet.from_name(arch, **kwargs)
	return model

def efficientnet_b0(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b0', pretrained, **kwargs)

def efficientnet_b1(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b1', pretrained, **kwargs)

def efficientnet_b2(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b2', pretrained, **kwargs)

def efficientnet_b3(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b3', pretrained, **kwargs)

def efficientnet_b4(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b4', pretrained, **kwargs)

def efficientnet_b5(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b5', pretrained, **kwargs)

def efficientnet_b6(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b6', pretrained, **kwargs)

def efficientnet_b7(pretrained=False, **kwargs):
	return _EfficientNet('efficientnet-b7', pretrained, **kwargs)