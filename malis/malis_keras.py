import keras.backend as K
import tensorflow as tf
import numpy as np
from .wrappers import get_pairs
from .pairs_cython import mknhood3d


class Malis:
    def __init__(self,
                 margin=0.3,
                 pos_loss_weight=0.3):
        self.margin = margin
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = 1 - pos_loss_weight

    def pairs_to_loss_keras(self, pos_pairs, neg_pairs, pred):
        """
        Input:
            pos_pairs: (batch_size, H, W, C)
               Contains the positive pairs stacked 
            neg_pairs: (batch_size, H, W, C)
               Contains the negative pairs 
            pred:  (batch_size, H, W, C)
                affinity predictions from network
        Returns:
            malis_loss: scale
        """
        zeros_helpervar = tf.zeros(shape=tf.shape(pred))

        pos_loss = tf.where(1 - pred - self.margin > 0,
                            (1 - pred - self.margin)**2,
                            zeros_helpervar)
        pos_loss = pos_loss * pos_pairs
        pos_loss = tf.reduce_sum(pos_loss) * self. pos_loss_weight

        neg_loss = tf.where(pred - self.margin > 0,
                            (pred - self.margin)**2,
                            zeros_helpervar)
        neg_loss = neg_loss * neg_pairs
        neg_loss = tf.reduce_sum(neg_loss) * self. neg_loss_weight
        malis_loss = (pos_loss + neg_loss) * 2  # because of the pos_loss_weight and neg_loss_weight

        return malis_loss

    def pairs_to_loss_python(self, pos_pairs,neg_pairs, pred):
        """
        Input:
        pos_pairs: (batch_size, H, W, C)
               Contains the positive pairs stacked 
        neg_pairs: (batch_size, H, W, C)
           Contains the negative pairs 
        pred:  (batch_size, H, W, C)
            affinity predictions from network

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

        pos_loss = np.where(1 - pred - self.margin > 0,
                            (1 - pred - self.margin)**2,
                            0.0)
        pos_loss = pos_loss * pos_pairs
        pos_loss = np.sum(pos_loss, axis=(1, 2, 3, 4)) * self. pos_loss_weight

        neg_loss = np.where(pred - self.margin > 0,
                            (pred - self.margin)**2,
                            0.0)
        neg_loss = neg_loss * neg_pairs
        neg_loss = np.sum(neg_loss, axis=(1, 2, 3, 4)) * self. neg_loss_weight
        malis_loss = (pos_loss + neg_loss) * 2

        return malis_loss, pos_loss, neg_loss

malis_obj = Malis()
def malis_loss(batchsize):
    def loss(y_true,y_pred): 
        '''
        Input:
            y_true: Tensor (batch_size, H, W, C = 1)
               segmentation groundtruth
            y_pred: Tensor (batch_size, H, W, C = 3/2)
                affinity predictions from network
        Returns:
            malis loss
        '''

        y = K.int_shape(y_pred)[1]  # H
        x = K.int_shape(y_pred)[2]  # W

        seg_true = K.reshape(y_true,(y,x,batchsize))   # (H,W,C'=C*batch_size)
        y_pred = K.permute_dimensions(y_pred,(3,1,2,0))   #(C=3/2,H,W,batch_size)
                
        '''
        malis loss part:
        nhood = mknhood3d(1)[:-1] #[[-1,0,0],[0,-1,0]] for 2d image 
        nhood = mknhood3d(1)      #[[-1,0,0],[0,-1,0],[0,0,-1]] for 3d image
        get_pairs: get positive and negtive weights
                   inputs: seg_true (H,W,C')
                           y_pred (edge,H,W,C')
                   outputs: pos_pairs and neg_pairs (edge,H,W,C')
        pairs_to_loss_keras: get final malis loss
                   inputs: pos_pairs,neg_paris: output of get_pairs (edge,H,W,C')
                           y_pred: (edge,H,W,C')
                   outputs: loss (scale tensor)
        '''
        nhood = mknhood3d(1)[:-1] 
        pos_pairs, neg_pairs = tf.numpy_function(func = get_pairs,inp=[seg_true, y_pred, nhood],
                                                 Tout=[tf.uint64,tf.uint64])
        pos_pairs = tf.cast(pos_pairs,tf.float32)
        neg_pairs = tf.cast(neg_pairs,tf.float32) 

        loss = malis_obj.pairs_to_loss_keras(pos_pairs, neg_pairs, y_pred)
        return loss
    
    return loss
