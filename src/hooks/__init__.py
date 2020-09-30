from .hooks_core import *
from .swa import SWA

def build_hooks(config):
	hooks = [Logging(), Validation()]
	return hooks