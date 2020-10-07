import torch
from ..models import build_model

def load_from_path(path):
	data = torch.load(path)
	params = data['params']
	config = data['cfg']
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = build_model(config)
	model.load_state_dict(params)
	print("Model loaded from: {}".format(path))
	return model