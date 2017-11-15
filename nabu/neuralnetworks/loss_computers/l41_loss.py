'''@file l41_loss.py
contains the L41Loss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops
import pdb

class L41Loss(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, logits,seq_length):
        '''
        Compute the loss

        Creates the operation to compute the Lab41 loss

        Args:
            targets: a dictionary of [batch_size x time x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x ? x ...] tensors containing the logits
            seq_length: a dictionary of [batch_size] vectors containing
                the sequence lengths

        Returns:
            loss: a scalar value containing the loss
            norm: a scalar value indicating how to normalize the loss
        '''
              
	binary_target=targets['binary_targets']            
	usedbins = targets['usedbins']
	seq_length = seq_length['bin_emb']
	bin_embeddings = logits['bin_emb']
	spk_embeddings = logits['spk_emb']
		    
	loss, norm = ops.L41_loss(binary_target, bin_embeddings, spk_embeddings,
			   usedbins, seq_length,self.batch_size)
            
        return loss, norm
