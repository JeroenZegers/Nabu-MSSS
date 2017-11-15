'''@file pit_l41_loss.py
contains the PITL41Loss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class PITL41Loss(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, logits, seq_length):
        '''
        Compute the loss

        Creates the operation to compute a loss that is the combination of Permudation 
        Invariant Training and L41 loss

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
                       
	multi_targets=targets['multi_targets']            
	mix_to_mask = targets['mix_to_mask']
	seq_length = seq_length['bin_emb']
	bin_embeddings = logits['bin_emb']
	spk_embeddings = logits['spk_emb']
		    
	loss, norm = ops.pit_L41_loss(multi_targets, bin_embeddings, spk_embeddings, mix_to_mask, 
					seq_length,self.batch_size)
            
        return loss, norm
