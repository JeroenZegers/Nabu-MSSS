'''@file deepclustering_2and3spk_loss.py
contains the DeepclusteringFull2and3SpkLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class Deepclustering2and3SpkLoss(loss_computer.LossComputer):
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
                       
	binary_target_2spk=targets['binary_targets_2spk']            
	usedbins_2spk = targets['usedbins_2spk']
	seq_length_2spk = seq_length['bin_emb_2spk']
	logits_2spk = logits['bin_emb_2spk']
		    
	loss_2spk, norm_2spk = ops.deepclustering_loss(binary_target_2spk, logits_2spk, usedbins_2spk, 
					seq_length_2spk,self.batch_size)
	
	binary_target_3spk=targets['binary_targets_3spk']            
	usedbins_3spk = targets['usedbins_3spk']
	seq_length_3spk = seq_length['bin_emb_3spk']
	logits_3spk = logits['bin_emb_3spk']
		    
	loss_3spk, norm_3spk = ops.deepclustering_loss(binary_target_3spk, logits_3spk, usedbins_3spk, 
					seq_length_3spk,self.batch_size)
            
	loss = loss_2spk + loss_3spk
	norm = norm_2spk + norm_3spk
	
        return loss, norm
