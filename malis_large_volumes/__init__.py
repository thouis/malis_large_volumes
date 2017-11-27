import numpy as np
from . import malis_keras, balanced_malis_keras
from .wrappers import get_pairs, get_pairs_python

# create alias for cython (cython is the default)
get_pairs_cython = get_pairs
