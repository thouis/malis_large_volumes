import numpy as np
import malis_large_volumes
import malis_large_volumes.malis_keras as mk
import tensorflow as tf

na = np.newaxis

#######################################################
# TEST 1
print("Starting test 1")
# the following variables are named the same as the ones in
# Turaga et al. 2009 (Malis paper)
x = 0.4
x_hat = 0.0
m = 0.3
pos_loss_weight = 0.5
neg_loss_weight = 0.5

labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.uint32)
affinities = np.zeros(shape=(3,) + labels.shape)
affinities[2, :, :, :] = np.array([[[.6, .6, .6, x, .6, .6]]], dtype=np.float64)
affinities += np.random.normal(size=affinities.shape, scale=.0001)

pos_pairs, neg_pairs = malis_large_volumes.get_pairs(labels, affinities,
                                                     stochastic_malis_param=0)
all_pairs = np.concatenate((pos_pairs, neg_pairs), axis=0).astype(np.float32)
malis_obj = malis_large_volumes.malis_keras.Malis(pos_loss_weight=0.5)

pairs_var = tf.placeholder(tf.float32, name="pairs")
affinities_var = tf.placeholder(tf.float32, name="affinities")
loss_var = malis_obj.pairs_to_loss_keras(pairs_var, affinities_var)
affinities_grad = tf.gradients(loss_var, [affinities_var])[0]

affinities = affinities[na, :]
all_pairs = all_pairs[na, :]
with tf.Session() as sess:
    loss = sess.run(loss_var, feed_dict={pairs_var: all_pairs,
                                         affinities_var: affinities})
    gradient = sess.run(affinities_grad, feed_dict={pairs_var: all_pairs,
                                                    affinities_var: affinities})
print("\n\n")

# use python computation:
python_loss, _, _ = malis_obj.pairs_to_loss_python(all_pairs, affinities)
python_loss = python_loss[0]

# analytically expected loss:
n_edges = affinities[0].size
pos_loss = pos_loss_weight * ((3 + 3) * (1 - 0.6) ** 2)
neg_loss = neg_loss_weight * ((3 * 3) * (x) ** 2)
analytic_loss = (pos_loss + neg_loss) * 2 / n_edges
analytic_gradient = 2 * 9 * (0.4) / n_edges

print("The keras/tensorflow computed loss is: " + str(loss[0]))
print("The python computed loss is: " + str(python_loss))
print("The analytically expected loss is: " + str(analytic_loss))
assert np.isclose(analytic_loss, loss, atol=0.1), "analytic loss and tensorflow loss not the same"
assert np.isclose(analytic_loss, python_loss, atol=0.1)

gradient = gradient[0, 2, 0, 0, 3]
print("The keras/tensorflow computed gradient at the 'border' affinity is: " + str(gradient))
print("The analytically expected gradient is: " + str(analytic_gradient))
assert np.isclose(analytic_gradient, gradient, atol=0.1)

print("Test 1 finished, no error")


########################################################
## TEST 2
print("\nTest 2")

# setting the 'border'-affinity to 0.1, which is below the margin, so there should be
# no loss for non-matching pairs and the gradient at this edge should be zero
affinities[0, 2, 0, 0, 3] = 0.1
with tf.Session() as sess:
    loss = sess.run(loss_var, feed_dict={pairs_var: all_pairs,
                                         affinities_var: affinities})
    gradient = sess.run(affinities_grad, feed_dict={pairs_var: all_pairs,
                                                    affinities_var: affinities})

# use python computation:
python_loss, _, _ = malis_obj.pairs_to_loss_python(all_pairs, affinities)
python_loss = python_loss[0]

# analytically expected loss:
n_edges = affinities.size
analytic_loss = (3 + 3) * (1 - 0.6) ** 2 + 0
analytic_loss /= n_edges
analytic_gradient = 0

print("The keras/tensorflow computed loss is: " + str(loss[0]))
print("The python computed loss is: " + str(python_loss))
print("The analytically expected loss is: " + str(analytic_loss))
assert np.isclose(analytic_loss, loss, atol=0.1)
assert np.isclose(analytic_loss, python_loss, atol=0.1)

gradient = gradient[0, 2, 0, 0, 3]
print("The keras/tensorflow computed gradient at the 'border' affinity is: " + str(gradient))
print("The analytically expected gradient is: " + str(analytic_gradient))
assert np.isclose(analytic_gradient, gradient, atol=0.1)

print("Test 2 finished, no error")
