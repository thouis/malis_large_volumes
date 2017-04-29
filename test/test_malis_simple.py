import numpy as np
import malis_large_volumes
import malis_large_volumes.malis_keras as mk
from malis_large_volumes import pairs_cython as pairs_cython
import malis.malis_pair_wrapper as malis_pairs_wrapper_turaga
import tensorflow as tf

na = np.newaxis

#######################################################
# TEST 1
print("Starting test 1")
labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.uint32)
affinities = np.zeros(shape=(3,) + labels.shape)
affinities[2, :, :, :] = np.array([[[.7, .7, .7, .3, .7, .7]]], dtype=np.float64)
affinities += np.random.normal(size=affinities.shape, scale=.0001)

pos_pairs, neg_pairs = malis_large_volumes.get_pairs(labels, affinities)
all_pairs = np.concatenate((pos_pairs, neg_pairs), axis=0).astype(np.float32)
malis = mk.Malis(pos_loss_weight=0.5)

pairs_var = tf.placeholder(tf.float32, name="pairs")
affinities_var = tf.placeholder(tf.float32, name="affinities")
loss_var = malis.pairs_to_loss_keras(pairs_var, affinities_var)
affinities_grad = tf.gradients(loss_var, [affinities_var])[0]

affinities = affinities[na, :]
all_pairs = all_pairs[na, :]
with tf.Session() as sess:
    loss = sess.run(loss_var, feed_dict={pairs_var: all_pairs,
                                  affinities_var: affinities})
    gradient = sess.run(affinities_grad, feed_dict={pairs_var: all_pairs,
                                                    affinities_var: affinities})


# analytically expected loss:
analytic_loss = (3 + 3) * (1-0.7) ** 2 + 9 * (0.3 ** 2)
analytic_gradient = 2 * 9 * 0.3

print("The computed loss is: " + str(loss[0]))
print("The analytically expected loss is: " + str(analytic_loss))
assert np.isclose(analytic_loss, loss, atol=0.1)

gradient = gradient[0, 2, 0, 0, 3]
print("The computed gradient at the 'border' affinity is: " + str(gradient))
print("The analytically expected gradient is: " + str(analytic_gradient))
assert np.isclose(analytic_gradient, gradient, atol=0.1)


print("Test 1 finished, no error")


########################################################
## TEST 2
#print("Starting test 2")
#
#labels = np.ones((4, 3, 3), dtype=np.uint32)
#labels[2:] = 2
#affinities = np.random.normal(size=(3,) + labels.shape, loc=.5, scale=.01).astype(np.float)
#affinities[0, 2, :, :] -= .3
#affinities[0, 2, 1, 1]  = .4 # this is the maximin edge between the two objects
#
#edge_tree = pairs_cython.build_tree(labels, affinities, neighborhood)
#pos_pairs, neg_pairs = pairs_cython.compute_pairs_with_tree(labels, affinities, neighborhood, edge_tree)
#assert neg_pairs[0, 2, 1, 1] == (2 * 3 * 3) ** 2
#
## compare with turagas implementation
#pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(affinities, 
#                                                         labels.astype(np.int64),
#                                                         ignore_background=False)
#assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
#assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
#print("Test 2 finished, no error")
#
#
########################################################
## TEST 3
#print("Starting test 3")
## in this test we're just comparing the current implementation and Turagas
#labels = np.random.randint(0, 10, size=(10, 20, 20), dtype=np.uint32)
#affinities = np.random.normal(loc=0.5, scale=0.1, size=(3,) + labels.shape).astype(np.float)
#
#edge_tree = pairs_cython.build_tree(labels, affinities, neighborhood)
#pos_pairs, neg_pairs = pairs_cython.compute_pairs_with_tree(labels, affinities, neighborhood, edge_tree, keep_objs_per_edge=20)
#
## compare with turagas implementation
#pos_pairs_2, neg_pairs_2 = malis_pairs_wrapper_turaga.get_counts(affinities, 
#                                                         labels.astype(np.int64),
#                                                         ignore_background=False)
#try:
#    assert np.all(pos_pairs == pos_pairs_2), "pos pairs was not same as turaga implementation"
#    assert np.all(neg_pairs == neg_pairs_2), "neg pairs was not same as turaga implementation"
#    print("Test 3 finished, no error")
#except:
#    print("Test 3 FAILED!")
#    print("Tree-malis was not the same as Turaga-malis.")
#    print("However, this happens sometimes and I assume it's due to differences in " + \
#          "sorting the edges. Try running the tests again and see if it fails again.")
#
########################################################
#
#
########################################################
## TEST 4
#print("Starting test 4")
## In this test we're testing the wrapper that will be used by external users
#pos_pairs_from_wrapper, neg_pairs_from_wrapper = malis_large_volumes.get_pairs(\
#        labels, affinities, neighborhood, keep_objs_per_edge=20)
#
#assert np.all(pos_pairs == pos_pairs_from_wrapper), "pos pairs was not same as pos pairs from wrapper"
#assert np.all(neg_pairs == neg_pairs_from_wrapper), "neg pairs was not same as neg pairs from wrapper"
#print("Test 4 finished, no error")
########################################################
