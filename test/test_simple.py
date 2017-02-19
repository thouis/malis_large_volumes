import numpy as np
from malis_large_volumes import pairs_python, pairs_cython
from malis_large_volumes import malis_python, malis_cython

labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.uint32)
weights = np.zeros(shape=labels.shape + (3,))
weights[:, :, :, 0] = np.array([[[.2, .2, .3, .2, .2, .2]]], dtype=np.float)


neighborhood = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.int32)

edge_tree = malis_cython.build_tree(labels, weights, neighborhood)
pos_pairs, neg_pairs = malis_cython.compute_pairs(labels, weights, neighborhood, edge_tree)
print("Neg pairs:")
print(neg_pairs)
assert neg_pairs[0, 0, 2, 2] == 9, "neg pairs result was incorrect"

print("Test finished, no error")


