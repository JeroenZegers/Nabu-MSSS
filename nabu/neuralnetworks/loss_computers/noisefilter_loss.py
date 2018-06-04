'''@file noisefilter_loss.py
contains the NoisefilterLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class NoisefilterLoss(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, logits, seq_length):
	'''
	    Compute the loss

	    Creates the operation to compute the loss of the noise filtering network

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

        clean_spectrogram=targets['cleanspectrogram']
        noise_spectrogram = targets['noisespectrogram']
        seq_length = seq_length['alpha']
        alpha = logits['alpha']


        loss, norm = ops.noise_filter_loss(clean_spectrogram,noise_spectrogram,
            alpha,seq_length,self.batch_size)


        return loss, norm
