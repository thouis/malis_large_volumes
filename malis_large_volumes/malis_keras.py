import keras.backend as K
import tensorflow as tf
import numpy as np


class Malis:
    def __init__(self,
                 keep_objs_per_edge=20,
                 margin=.2,
                 pos_loss_weight=0.5):
        self.margin = margin
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = 1 - pos_loss_weight

    def pairs_to_loss_keras(self, pairs, pred):
        """
        NOTE: right now this is not theano compatible (TODO)
        Input:
        pairs [batch_size, 2*K, D, W, H]
               Contains the positive and negative pairs stacked on top of each
               other in the second dimension
        pred  [batch_size, K, D, W, H]
        Returns:
        malis_loss, vector of shape (batch_size,)
        """
        pos_pairs = pairs[:, 0:3]
        neg_pairs = pairs[:, 3:6]

        # compute loss mask for losses outside of margin
        ones_helpervar = tf.ones(shape=tf.shape(pred))
        zeros_helpervar = tf.zeros(shape=tf.shape(pred))
        edges_not_in_upper_margin = tf.where(pred < 1 - self.margin, ones_helpervar, zeros_helpervar)
        edges_not_in_lower_margin = tf.where(pred > self.margin, ones_helpervar, zeros_helpervar)
        loss_mask = tf.where(pos_pairs > neg_pairs, edges_not_in_upper_margin, edges_not_in_lower_margin)

        pos_loss = (1 - pred)**2 * pos_pairs * loss_mask * self.pos_loss_weight
        neg_loss = pred**2 * neg_pairs * loss_mask * self.neg_loss_weight
        # get the total loss at each element
        elemwise_total_loss = pos_loss + neg_loss
        malis_loss = K.sum(elemwise_total_loss, axis=(1, 2, 3, 4))
        malis_loss = malis_loss * 2  # because of the pos_loss_weight and neg_loss_weight
        return malis_loss

    def pairs_to_loss_python(self, pairs, pred):
        """
        Input:
        pairs [batch_size, 2*K, D, W, H]
              pairs contains both the matching and nonmatching pairs.
              On axis 1, the first three slices are for the matching pairs
              and the last three for the nonmatching pairs (there are always
              3 for the three directions in which there are affinities)
        pred [batch_size, K, D, W, H]

        Returns:
        malis_loss, vector of shape (batch_size,)
        pos_loss, vector of shape (batch_size,)
        neg_loss, vector of shape (batch_size,)
        """
        # the code is deliberately kept very similar to pairs_to_loss_keras
        # in order to verify that the keras version does the right thing
        # (more performant code may very well be possile but that is not the goal)

        pos_pairs = pairs[:, 0:3]
        neg_pairs = pairs[:, 3:6]

        # compute loss mask for losses outside of margin
        ones_helpervar = np.ones(shape=np.shape(pred))
        zeros_helpervar = np.zeros(shape=np.shape(pred))
        edges_not_in_upper_margin = np.where(pred < 1 - self.margin, ones_helpervar, zeros_helpervar)
        edges_not_in_lower_margin = np.where(pred > self.margin, ones_helpervar, zeros_helpervar)
        loss_mask = np.where(pos_pairs > neg_pairs, edges_not_in_upper_margin, edges_not_in_lower_margin)

        pos_loss = (1 - pred)**2 * pos_pairs * loss_mask * self.pos_loss_weight
        neg_loss = pred**2 * neg_pairs * loss_mask * self.neg_loss_weight
        # get the total loss at each element
        elemwise_total_loss = pos_loss + neg_loss
        malis_loss = np.sum(elemwise_total_loss, axis=(1, 2, 3, 4)) * 2
        pos_loss = np.sum(pos_loss, axis=(1, 2, 3, 4)) * 2
        neg_loss = np.sum(neg_loss, axis=(1, 2, 3, 4)) * 2
        return malis_loss, pos_loss, neg_loss
