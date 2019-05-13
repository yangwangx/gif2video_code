from .misc import *
from .files import *
from .torch_funcs import *

# import subdir as modules
def _add_cwd():
    import os, sys
    try:
        cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
    except NameError:
        cwd = ''
    sys.path.append(cwd + './')
_add_cwd()
import torch_data as torchdata
import torch_model as torchmodel
