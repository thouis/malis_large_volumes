import numpy as np
#from . import malis_keras
#from . import malis_torch
from .wrappers import get_pairs, get_pairs_python
from .pairs_cython import seg_to_affgraph,mknhood3d,affgraph_to_seg


## using keras: from malis.malis_keras import malis_loss:   loss = malis_loss(seg_gt,aff_pred)
## using pytorch: from malis.malis_torch import malis_loss: loss = malis_loss(aff_pred,seg_gt)