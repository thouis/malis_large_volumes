import numpy as np
from malis_large_volumes import pairs_python, pairs_cython
from malis_large_volumes import malis_python, malis_cython
from test_tools import malis_turaga

labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.uint32)
weights = np.zeros(shape=labels.shape + (3,))
weights[:, :, :, 2] = np.array([[[.2, .2, .2, .1, .2, .2]]], dtype=np.float) 
weights += np.random.normal(size=weights.shape, scale=.001)


neighborhood = np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=np.int32)

edge_tree = malis_cython.build_tree(labels, weights, neighborhood)
pos_pairs, neg_pairs = malis_cython.compute_pairs(labels, weights, neighborhood, edge_tree)
assert neg_pairs[0, 0, 3, 2] == 9, "neg pairs result was incorrect"

########################################################
# compare with turagas implementation
pos_pairs_2, neg_pairs_2 = malis_turaga(weights, labels)
assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"

print("Test finished, no error")
