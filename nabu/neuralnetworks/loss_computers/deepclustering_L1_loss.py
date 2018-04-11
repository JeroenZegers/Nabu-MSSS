'''@file deepclustering_loss.py
contains the DeepclusteringLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class DeepclusteringL1Loss(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, logits, seq_length):
        '''
        Compute the loss

        Creates the operation to compute the deep clustering loss with L1 norm

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
	usedbins = targets['usedbins']
	seq_length = seq_length['bin_emb']
	logits = logits['bin_emb']
		    
	loss, norm = ops.deepclustering_L1_loss(binary_target, logits, usedbins, 
					seq_length,self.batch_size)
            
        return loss, norm
