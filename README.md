# Malis Loss

## What is this?
This program MALIS loss described in for 2D and 3D data:

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). *Maximin learning of image segmentation*. _Advances in Neural Information Processing Systems (NIPS) 2009_.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation


## Installation:
```
./make.sh            (Building c++ extension only: run inside directory)
pip install .        (Installation as python package: run inside directory)
```


### Installation example in anaconda:
```
conda create -n malis python=3.7
conda install cython
conda install numpy
conda install gxx_linux-64
conda install -c anaconda boost
./make.sh
pip install .
```

## Example Usage:

### Using Keras/Tensorflow (channel last):

#### 2D usage
```
import malis as m
from malis.malis_keras import pairs_to_loss_keras

def malis_loss(y_true,y_pred): 
    # Input:
    #    y_true: Tensor (batch_size, H, W, C = 1)
    #       segmentation groundtruth
    #    y_pred: Tensor (batch_size, H, W, C = 2)
    #        affinity predictions from network
    # Returns:
    #    loss: Tensor(scale)
    #           malis loss 
    
    ######### please modify here to make sure seg_true and y_pred has the correct shape      
    x = K.int_shape(y_pred)[1]  # H
    y = K.int_shape(y_pred)[2]  # W

    seg_true = K.reshape(y_true,(x,y,-1))             # (H,W,C'=C*batch_size)
    y_pred = K.permute_dimensions(y_pred,(3,1,2,0))   # (C=2,H,W,batch_size)
    #########
    
    nhood = malis.mknhood3d(1)[:-1]                    
    pos_pairs, neg_pairs = tf.numpy_function(func = malis.get_pairs,inp=[seg_true, y_pred, nhood],
                                             Tout=[tf.uint64,tf.uint64])
    pos_pairs = tf.cast(pos_pairs,tf.float32)
    neg_pairs = tf.cast(neg_pairs,tf.float32) 

    loss = pairs_to_loss_keras(pos_pairs, neg_pairs, y_pred)
    
    return loss

model = ... (set the channel of output layer as 2)

model.compile(optimizer, loss = malis_loss)
```

#### 3D usage (please use batch size as 1)
```
import malis as m
from malis.malis_keras import pairs_to_loss_keras

def malis_loss(y_true,y_pred): 
    # Input:
    #    y_true: Tensor (batch_size=1, H, W, D, C=1)
    #       segmentation groundtruth
    #    y_pred: Tensor (batch_size=1, H, W, D, C=3)
    #        affinity predictions from network
    # Returns:
    #    loss: Tensor(scale)
    #           malis loss 
    
    ######### please modify here to make sure seg_true and y_pred has the correct shape      
    x = K.int_shape(y_pred)[1]  # H
    y = K.int_shape(y_pred)[2]  # W
    z = K.int_shape(y_pred)[3]  # D

    seg_true = K.reshape(y_true,(x,y,z))              # (H,W,D)
    y_pred = K.reshape(y_pred,(H,W,D,-1))             # (H,W,D,C=3)
    y_pred = K.permute_dimensions(y_pred,(3,0,1,2))   # (C=3,H,W,D)
    
    #########
    
    nhood = malis.mknhood3d(1)                  
    pos_pairs, neg_pairs = tf.numpy_function(func = malis.get_pairs,inp=[seg_true, y_pred, nhood],
                                             Tout=[tf.uint64,tf.uint64])
    pos_pairs = tf.cast(pos_pairs,tf.float32)
    neg_pairs = tf.cast(neg_pairs,tf.float32) 

    loss = pairs_to_loss_keras(pos_pairs, neg_pairs, y_pred)
    
    return loss

model = ... (set the channel of output layer as 3)

model.compile(optimizer, loss = malis_loss)
```

### Using Pytorch: 
#### 2D usage
```
import malis
from malis.malis_torch import torchloss,pairs_to_loss_torch

def malis_loss(seg_gt,output): 
    
    # Input:
    #    output: Tensor(batch size, channel=2, H, W)
    #           predicted affinity graphs from network
    #    seg_gt: Tensor(batch size, channel=1, H, W)
                segmentation groundtruth     
    # Returns: 
    #    loss: Tensor(scale)
    #           malis loss 
    
    ######### please modify here to make sure seg_gt and output has the correct shape
    x,y = seg_gt.shape[2],seg_gt.shape[3]
    output = output.permute(1,2,3,0)           # (2,H,W,batch_size)
    seg_gt = seg_gt.reshape(x,y,-1)            # (H,W,C'=C*batch_size)
    #########
    
    nhood = malis.mknhood3d(1)[:-1]  
    pos_pairs,neg_pairs = torchloss.apply(output, seg_gt, nhood)
    loss = pairs_to_loss_torch(pos_pairs, neg_pairs, output)
    
    return loss
    
loss = malis_loss(seg_gt, output)
```
#### 3D usage (please use batch size as 1)
```
import malis
from malis.malis_torch import torchloss,pairs_to_loss_torch

def malis_loss(seg_gt,output): 
    
    # Input:
    #    output: Tensor(batch size=1, channel=3, H, W, D)
    #           predicted affinity graphs from network
    #    seg_gt: Tensor(batch size=1, channel=1, H, W, D)
                segmentation groundtruth     
    # Returns: 
    #    loss: Tensor(scale)
    #           malis loss 
    
    ######### please modify here to make sure seg_gt and output has the correct shape
    x,y,z = seg_gt.shape[2],seg_gt.shape[3],seg_gt.shape[4]
    output = output.reshape(-1,x,y,z)         # (3,H,W,D)
    seg_gt = seg_gt.reshape(x,y,z)            # (H,W,D)
    #########
    
    nhood = malis.mknhood3d(1) 
    pos_pairs,neg_pairs = torchloss.apply(output, seg_gt, nhood)
    loss = pairs_to_loss_torch(pos_pairs, neg_pairs, output)
    
    return loss
    
loss = malis_loss(seg_gt, output)
```

### Useful Functions of malis loss in python:
```
import malis as m
nhood = m.mknhood3d(): Makes neighbourhood structures
aff = m.seg_to_affgraph(seg_gt,nhood): Construct an affinity graph from a segmentation
seg = m.affgraph_to_seg(affinity,nhood): Obtain a segentation graph from an affinity graph
```
### Postprocessing:
The output of network should be affinity graphs, to obtain final segmentation graphs, threshold should be manully selected and than apply affgraph_to_seg functions. An example is like below:
```
import malis as m
import numpy as np

aff = .... # predicted affinity graph from trained model
aff = np.where(aff<threshold,0,1)
nhood = malis.mknhood3d(1)[:-1]  # or malis.mknhood3d(1) for 3d data prediction
seg = m.affgraph_to_seg(aff,nhood)
```