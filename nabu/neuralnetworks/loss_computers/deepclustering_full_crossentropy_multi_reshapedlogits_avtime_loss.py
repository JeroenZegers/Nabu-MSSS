'''@file deepclustering_full_crossentropy_multi_reshapedlogits_avtime_loss.py
contains the DeepclusteringFullCrossEntropyMultiReshapedLogitsAvTimeLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class DeepclusteringFullCrossEntropyMultiReshapedLogitsAvTimeLoss(loss_computer.LossComputer):
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
	
	alpha=1e-1
	
	#dc loss
	binary_target=targets['binary_targets']            
	usedbins = targets['usedbins']
	seq_length = seq_length['bin_emb']
	logits_dc = logits['bin_emb']
		    
	loss_dc, norm_dc = ops.deepclustering_full_loss_efficient(binary_target, logits_dc, usedbins, 
					seq_length,self.batch_size)
	
	#cross-entropy loss
	spkids=targets['spkids']        
	logits_cro = logits['spkest']
	
	nrS = spkids.get_shape()[1]

	logits_cro=tf.reduce_mean(logits_cro,1)
	logits_cro=tf.reshape(logits_cro,[self.batch_size,nrS,-1])
		    
	loss_cro, norm_cro = ops.crossentropy_multi_loss(spkids, logits_cro, self.batch_size)
	
	loss = loss_dc/norm_dc + alpha * loss_cro/norm_cro
	norm = 1.0
            
        return loss, norm
