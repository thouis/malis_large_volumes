import numpy as np
from malis_large_volumes import pairs_python, pairs_cython

labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.int32)
weights = np.zeros(shape=labels.shape + (3,))
weights[:, :, :, 0] = np.array([[[.2, .2, .3, .2, .2, .2]]], dtype=np.float)

pos_pairs, neg_pairs = pairs_python(labels, weights)
import pdb; pdb.set_trace()


