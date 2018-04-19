'''@file deepclusteringnoise_loss.py
contains the DeepclusteringnoiseLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class NoisefilterLoss(loss_computer.LossComputer):
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

        clean_spectrogram=targets['cleanspectrogram']
        noise_spectrogram = targets['noisespectrogram']
        seq_length = seq_length['noise_filter']
        noise_filter = logits['noise_filter']


        loss, norm = ops.noise_filter_loss(clean_spectrogram,noise_spectrogram,
            noise_filter,seq_length,self.batch_size)
        
        
        return loss, norm
