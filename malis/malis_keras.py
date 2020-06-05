import keras.backend as K
import tensorflow as tf
import numpy as np
from .wrappers import get_pairs



def pairs_to_loss_keras(pos_pairs, neg_pairs, pred, margin=0.3, pos_loss_weight=0.3):
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
    neg_loss_weight = 1 - pos_loss_weight
    zeros_helpervar = tf.zeros(shape=tf.shape(pred))

    pos_loss = tf.where(1 - pred - margin > 0,
                        (1 - pred - margin)**2,
                        zeros_helpervar)
    pos_loss = pos_loss * pos_pairs
    pos_loss = tf.reduce_sum(pos_loss) * pos_loss_weight

    neg_loss = tf.where(pred - margin > 0,
                        (pred - margin)**2,
                        zeros_helpervar)
    neg_loss = neg_loss * neg_pairs
    neg_loss = tf.reduce_sum(neg_loss) * neg_loss_weight
    malis_loss = (pos_loss + neg_loss) * 2  # because of the pos_loss_weight and neg_loss_weight

    return malis_loss



def malis_loss(y_true,y_pred): #(b,512,512,1) (b,512,512,3)
    '''
    Input:
        y_true: Tensor (batch_size, H, W, C = 1)
           segmentation groundtruth
        y_pred: Tensor (batch_size, H, W, C = 3/2)
            affinity predictions from network
    Returns:
    '''
        
    y = K.int_shape(y_pred)[1]  # H
    x = K.int_shape(y_pred)[2]  # W

    seg_true = K.reshape(y_true,(y,x,-1))   # (H,W,C'=C*batch_size)
    y_pred = K.permute_dimensions(y_pred,(3,1,2,0))   #(C=3,H,W,batch_size)

    pos_pairs, neg_pairs = tf.numpy_function(func = get_pairs,inp=[seg_true, y_pred],
                                             Tout=[tf.uint64,tf.uint64])
    pos_pairs = tf.cast(pos_pairs,tf.float32)
    neg_pairs = tf.cast(neg_pairs,tf.float32) 

    loss = pairs_to_loss_keras(pos_pairs, neg_pairs, y_pred)
    
    return loss