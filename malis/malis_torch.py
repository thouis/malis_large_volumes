import torch
from torch.autograd import Function
from .wrappers import get_pairs
import numpy as np


class torchloss(Function):
    @staticmethod
    def forward(ctx, aff_pred, seg_gt): #(b,2, 512,512) (b,1,512,512)
        aff_pred = aff_pred.detach().numpy()
        seg_gt = seg_gt.detach().numpy()
        
        x,y = seg_gt.shape[2],seg_gt.shape[3]
        aff_pred = np.transpose(aff_pred,(1,2,3,0))
        seg_gt = seg_gt.reshape(x,y,-1)
        
        
        weights_pos,weights_neg = get_pairs(seg_gt,aff_pred)
        weights_pos = weights_pos.astype(np.float32)
        weights_neg = weights_neg.astype(np.float32)
        
        weights_pos = np.transpose(weights_pos,(-1,0,1,2))
        weights_neg = np.transpose(weights_neg,(-1,0,1,2))

        return torch.FloatTensor(weights_pos),torch.FloatTensor(weights_neg)

    @staticmethod
    def backward(ctx, grad_output1,grad_output2):
        return None,None

def malis_loss(output, seg_gt,margin=0.3,pos_loss_weight=0.3):
    '''
    input - output, Tensor(batch size, channel, H, W): predicted affinity graphs from network
    input - seg_gt, Tensor(batch size, channel=1, H, W): segmentation groundtruth 
    
    output - loss, Tensor(scale): malis loss 
    '''
    neg_loss_weight = 1 - pos_loss_weight
    zeros_helpervar = torch.zeros(output.size())
    pos_pairs,neg_pairs = torchloss.apply(output, seg_gt)
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

