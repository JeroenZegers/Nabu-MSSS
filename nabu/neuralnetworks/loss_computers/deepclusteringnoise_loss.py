'''@file deepclusteringnoise_loss.py
contains the DeepclusteringnoiseLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class DeepclusteringnoiseLoss(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, logits, seq_length):
    '''
    Compute the loss

    Creates the operation to compute the deep clustering loss

    Args:
        targets: a dictionary of [batch_size x time x ...] tensor containing
            the targets
        logits: a dictionary of [batch_size x time x ...] tensors containing the logits
        seq_length: a dictionary of [batch_size] vectors containing
            the sequence lengths

    Returns:
        loss: a scalar value containing the loss
        norm: a scalar value indicating how to normalize the loss
    '''

        binary_target=targets['binary_targets']
        noise_target = targets['noise_targets'][:,:,nrS::nrS+1] 
        usedbins = targets['usedbins']
        seq_length = seq_length['bin_emb']
        emb_vec = logits['bin_emb']
        noise_detect = logits['noise_labels']
     
       
        loss, norm = ops.deepclustering_noise_loss(binary_target,noise_target, emb_vec,noise_detect, usedbins,
			        seq_length,self.batch_size)
            
        return loss, norm
