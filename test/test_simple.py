import numpy as np
from malis_large_volumes import pairs_python, pairs_cython
from malis_large_volumes import malis_python, malis_cython

labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.int32)
weights = np.zeros(shape=labels.shape + (3,))
weights[:, :, :, 0] = np.array([[[.2, .2, .3, .2, .2, .2]]], dtype=np.float)


neighborhood = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.int32)

edge_tree = malis_cython.build_tree(labels, weights, neighborhood)
pos_pairs, neg_pairs = malis_python.compute_pairs(labels, weights, neighborhood, edge_tree)
import pdb; pdb.set_trace()


