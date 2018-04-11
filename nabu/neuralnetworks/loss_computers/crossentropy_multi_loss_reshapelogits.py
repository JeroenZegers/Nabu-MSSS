'''@file crossentropy_multi_loss_reshapelogits.py
contains the CrossEntropyMultiLossReshapeLogits'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops
import pdb

class CrossEntropyMultiLossReshapeLogits(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, logits, seq_length=None):
        '''
        Compute the loss

        Creates the operation to compute the crossentropy multi loss

        Args:
            targets: a dictionary of [batch_size x ... x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x ... x ...] tensors containing the logits

        Returns:
            loss: a scalar value containing the loss
            norm: a scalar value indicating how to normalize the loss
        '''
                       
	spkids=targets['spkids']        
	logits = logits['spkest']
	
	nrS = spkids.get_shape()[1]
	#nrS = tf.shape(spkids)[1]

	logits=tf.reshape(logits,[self.batch_size,nrS,-1])
		    
	loss, norm = ops.crossentropy_multi_loss(spkids, logits, self.batch_size)
            
        return loss, norm
