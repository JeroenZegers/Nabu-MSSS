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

	    Creates the operation to compute the deep clustering loss when
        modified network architecture is used

	    Args:
	        targets: a dictionary of [batch_size x time x ...] tensor containing
	            the targets
        	logits: a dictionary of [batch_size x time x ...] tensors containing the logits
		seq_length: a dictionary of [batch_size] vectors containing
	            the sequence lengths

	    Returns:
        	loss: a scalar value containing the loss (of mini-batch)
	        norm: a scalar value indicating how to normalize the loss
	 '''

        binary_target=targets['binary_targets'] # partition targets
        energybins = targets['usedbins'] #  which bins contain enough energy
        seq_length = seq_length['bin_emb'] # Sequence length of utterances in batch
        emb_vec = logits['bin_emb'] # Embeddingvectors
        alpha = logits['noise_filter'] # alpha outputs
        nrS = tf.shape(binary_target)[2]/tf.shape(usedbins)[2] # Number of speakers in mixture
        noise_target = targets['noise_targets'][:,:,nrS::nrS+1] # Dominates noise the cell
        ideal_ratio = targets['ideal_ratio'] # Ideal ratio masker to filter noise
        # Calculate cost and normalisation
        loss, norm = ops.deepclustering_noise_loss(binary_target,noise_target,ideal_ratio,emb_vec,
            alpha,energybins,seq_length,self.batch_size)

        return loss, norm
