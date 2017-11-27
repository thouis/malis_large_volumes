import keras.backend as K
import tensorflow as tf
import numpy as np


class Malis:
    def __init__(self,
                 margin=0.3,
                 pos_loss_weight=0.3):
        self.margin = margin
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = 1 - pos_loss_weight

    def pairs_to_loss_keras(self, pairs, pred):
        """
        NOTE: This is not theano compatible
        Input:
            pairs: (batch_size, 2*K, D, W, H)
               Contains the positive and negative pairs stacked on top of each
               other in the second dimension
            pred:  (batch_size, K, D, W, H)
                affinity predictions
        Returns:
            malis_loss: vector of shape (batch_size,)
        """
        zeros_helpervar = tf.zeros(shape=tf.shape(pred))
        pos_pairs = pairs[:, 0:3]
        neg_pairs = pairs[:, 3:6]
        balanced_pairs = pos_pairs - neg_pairs

        pos_loss = tf.where(tf.logical_and(balanced_pairs > 0,
                                           1 - pred - self.margin > 0),
                            (1 - pred - self.margin)**2,
                            zeros_helpervar)
        pos_loss = pos_loss * tf.abs(balanced_pairs)
        pos_loss = K.sum(pos_loss, axis=(1, 2, 3, 4)) * self. pos_loss_weight

        neg_loss = tf.where(tf.logical_and(balanced_pairs <= 0,
                                           pred - self.margin > 0),
                            (pred - self.margin)**2,
                            zeros_helpervar)
        neg_loss = neg_loss * tf.abs(balanced_pairs)
        neg_loss = K.sum(neg_loss, axis=(1, 2, 3, 4)) * self. neg_loss_weight
        malis_loss = (pos_loss + neg_loss) * 2  # because of the pos_loss_weight and neg_loss_weight

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
        balanced_pairs = pos_pairs - neg_pairs

        pos_loss = np.where(np.logical_and(balanced_pairs > 0,
                                           1 - pred - self.margin > 0),
                            (1 - pred - self.margin)**2,
                            0.0)
        pos_loss = pos_loss * np.abs(balanced_pairs)
        pos_loss = np.sum(pos_loss, axis=(1, 2, 3, 4)) * self. pos_loss_weight

        neg_loss = np.where(np.logical_and(balanced_pairs <= 0,
                                           pred - self.margin > 0),
                            (pred - self.margin)**2,
                            0.0)
        neg_loss = neg_loss * np.abs(balanced_pairs)
        neg_loss = np.sum(neg_loss, axis=(1, 2, 3, 4)) * self. neg_loss_weight
        malis_loss = (pos_loss + neg_loss) * 2

        return malis_loss, pos_loss, neg_loss
