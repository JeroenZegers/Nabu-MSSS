'''@file dc_pit_loss.py
contains the DcPitLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class DcPitLoss(loss_computer.LossComputer):
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
	usedbins = targets['usedbins']               
	multi_targets=targets['multi_targets']            
	mix_to_mask = targets['mix_to_mask']
	seq_length = seq_length['bin_emb']
	logits_dc = logits['bin_emb']
	logits_pit = logits['bin_est']
	
	#rougly estimated loss scaling factor so PIT loss and DC loss are more or less of the same magnitude
	alpha=1.423024812840571e-09
		    
	loss, norm = ops.dc_pit_loss(binary_target, logits_dc,multi_targets, logits_pit,
					usedbins, mix_to_mask, seq_length,self.batch_size,alpha)
            
        return loss, norm
