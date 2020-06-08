import torch
from torch.autograd import Function
from .wrappers import get_pairs
import numpy as np
from .pairs_cython import mknhood3d

class torchloss(Function):
    @staticmethod
    def forward(ctx, aff_pred, seg_gt,nhood=None): #(b,2, 512,512) (b,1,512,512)
        aff_pred = aff_pred.detach().numpy()
        seg_gt = seg_gt.detach().numpy()       
        
        weights_pos,weights_neg = get_pairs(seg_gt,aff_pred,nhood)
        weights_pos = weights_pos.astype(np.float32)
        weights_neg = weights_neg.astype(np.float32)

        return torch.FloatTensor(weights_pos),torch.FloatTensor(weights_neg)

    @staticmethod
    def backward(ctx, grad_output1,grad_output2):
        return None,None,None

def pairs_to_loss_torch(pos_pairs, neg_pairs, output, margin=0.3, pos_loss_weight=0.3):
    '''
    input - output, Tensor(batch size, channel, H, W): predicted affinity graphs from network
    input - seg_gt, Tensor(batch size, channel=1, H, W): segmentation groundtruth 
    
    output - loss, Tensor(scale): malis loss 
    '''
    neg_loss_weight = 1 - pos_loss_weight
    zeros_helpervar = torch.zeros(output.size())

    pos_loss = torch.where(1 - output - margin > 0,
                        (1 - output - margin)**2,
                        zeros_helpervar)

    pos_loss = pos_loss * pos_pairs
    pos_loss = torch.sum(pos_loss) * pos_loss_weight

    neg_loss = torch.where(output - margin > 0,
                        (output - margin)**2,
                        zeros_helpervar)
    neg_loss = neg_loss * neg_pairs
    neg_loss = torch.sum(neg_loss) * neg_loss_weight
    loss = (pos_loss + neg_loss) * 2 
    
    return loss

def malis_loss2d(seg_gt,output): 
    
    # Input:
    #    output: Tensor(batch size, channel=2, H, W)
    #           predicted affinity graphs from network
    #    seg_gt: Tensor(batch size, channel=1, H, W)
    #            segmentation groundtruth     
    # Returns: 
    #    loss: Tensor(scale)
    #           malis loss 
    
    ######### please modify here to make sure seg_gt and output has the correct shape
    x,y = seg_gt.shape[2],seg_gt.shape[3]
    output = output.permute(1,2,3,0)           # (2,H,W,batch_size)
    seg_gt = seg_gt.reshape(x,y,-1)            # (H,W,C'=C*batch_size)
    #########
    
    nhood = mknhood3d(1)[:-1]  
    pos_pairs,neg_pairs = torchloss.apply(output, seg_gt, nhood)
    loss = pairs_to_loss_torch(pos_pairs, neg_pairs, output)
    
    return loss

def malis_loss3d(seg_gt,output): 
    
    # Input:
    #    output: Tensor(batch size=1, channel=3, H, W, D)
    #           predicted affinity graphs from network
    #    seg_gt: Tensor(batch size=1, channel=1, H, W, D)
    #            segmentation groundtruth     
    # Returns: 
    #    loss: Tensor(scale)
    #           malis loss 
    
    ######### please modify here to make sure seg_gt and output has the correct shape
    x,y,z = seg_gt.shape[2],seg_gt.shape[3],seg_gt.shape[4]
    output = output.reshape(-1,x,y,z)         # (3,H,W,D)
    seg_gt = seg_gt.reshape(x,y,z)            # (H,W,D)
    #########
    
    nhood = mknhood3d(1) 
    pos_pairs,neg_pairs = torchloss.apply(output, seg_gt, nhood)
    loss = pairs_to_loss_torch(pos_pairs, neg_pairs, output)
    
    return loss