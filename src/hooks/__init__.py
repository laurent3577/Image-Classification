from .hooks_core import *

def build_hooks(config):
	hooks = [Logging(), Validation()]
	return hooks