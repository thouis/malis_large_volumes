# Malis Loss

## What is this?
This program computes the object pair counts associated with the MALIS method as described in

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
```
import malis as m
from malis.malis_keras import pairs_to_loss_keras
```

model.compile(optimizer, loss = malis_loss(batch_size))

### Using Pytorch: 
```
import malis as m
from malis.malis_torch import torchloss,pairs_to_loss_torch
```

loss = malis_loss(aff_pred,seg_gt)


### Functions of malis loss in python:
```
nhood = m.mknhood3d(): Makes neighbourhood structures
seg = m.seg_to_affgraph(seg_gt,nhood): Construct an affinity graph from a segmentation
aff = m.affgraph_to_seg(affinity,nhood): Obtain a segentation graph from an affinity graph
```
