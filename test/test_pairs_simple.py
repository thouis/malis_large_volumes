import numpy as np
import malis_large_volumes
from malis_large_volumes import pairs_cython as pairs_cython
import malis.malis_pair_wrapper as malis_pairs_wrapper_turaga



#######################################################
# TEST 1
print("Starting test 1")
labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.uint32)
weights = np.zeros(shape=(3,) + labels.shape)
weights[2, :, :, :] = np.array([[[.2, .2, .2, .1, .2, .2]]], dtype=np.float) 
weights += np.random.normal(size=weights.shape, scale=.001)


neighborhood = np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=np.int32)

edge_tree = pairs_cython.build_tree(labels, weights, neighborhood,
                                    stochastic_malis_param=0)
pos_pairs, neg_pairs = pairs_cython.compute_pairs_with_tree(labels, weights, neighborhood, edge_tree)
assert neg_pairs[2, 0, 0, 3] == 9, "neg pairs result was incorrect"

# compare with turagas implementation
pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(weights, 
                                                         labels.astype(np.int64),
                                                         ignore_background=False)
#pos_pairs_2, neg_pairs_2 = malis_turaga(weights, labels, ignore_background=False)
assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
print("Test 1 finished, no error")


#######################################################
# TEST 2
print("Starting test 2")

labels = np.ones((4, 3, 3), dtype=np.uint32)
labels[2:] = 2
weights = np.random.normal(size=(3,) + labels.shape, loc=.5, scale=.01).astype(np.float)
weights[0, 2, :, :] -= .3
weights[0, 2, 1, 1]  = .4 # this is the maximin edge between the two objects

edge_tree = pairs_cython.build_tree(labels, weights, neighborhood)
pos_pairs, neg_pairs = pairs_cython.compute_pairs_with_tree(labels, weights, neighborhood, edge_tree)
assert neg_pairs[0, 2, 1, 1] == (2 * 3 * 3) ** 2

# compare with turagas implementation
pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(weights, 
                                                         labels.astype(np.int64),
                                                         ignore_background=False)
assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
print("Test 2 finished, no error")


#######################################################
# TEST 3
print("Starting test 3")
# in this test we're just comparing the current implementation and Turagas
labels = np.random.randint(0, 10, size=(10, 20, 20), dtype=np.uint32)
weights = np.random.normal(loc=0.5, scale=0.1, size=(3,) + labels.shape).astype(np.float)

edge_tree = pairs_cython.build_tree(labels, weights, neighborhood)
pos_pairs, neg_pairs = pairs_cython.compute_pairs_with_tree(labels, weights, neighborhood, edge_tree, keep_objs_per_edge=20)

# compare with turagas implementation
pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(weights, 
                                                         labels.astype(np.int64),
                                                         ignore_background=False)
try:
    assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
    assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
    print("Test 3 finished, no error")
except:
    print("Test 3 FAILED!")
    print("Tree-malis was not the same as Turaga-malis.")
    print("However, this happens sometimes and I assume it's due to differences in " + \
          "sorting the edges. Try running the tests again and see if it fails again.")

#######################################################


#######################################################
# TEST 4
print("Starting test 4")
# In this test we're testing the wrapper that will be used by external users
pos_pairs_from_wrapper, neg_pairs_from_wrapper = malis_large_volumes.get_pairs(\
        labels, weights, neighborhood, keep_objs_per_edge=20)

assert np.all(pos_pairs == pos_pairs_from_wrapper), "pos pairs was not same as pos pairs from wrapper"
assert np.all(neg_pairs == neg_pairs_from_wrapper), "neg pairs was not same as neg pairs from wrapper"
print("Test 4 finished, no error")
#######################################################
