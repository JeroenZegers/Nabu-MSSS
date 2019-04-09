'''@file deepattractornet_loss.py
contains the DeepattractornetSigmoidLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class DeepattractornetSigmoidLoss(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, logits, seq_length):
        '''
        Compute the loss

        Creates the operation to compute the deep attractor network with sigmoid mask loss

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

        # Which class belongs bin
        partioning = targets['partitioning']
        # Clean spectograms of sources
        spectrogram_targets=targets['spectogram_targets']
        # Spectogram of the original mixture
        mix_to_mask = targets['mix_to_mask']
        # Which bins contain enough energy
        energybins = targets['energybins']
        seq_length = seq_length['emb_vec']
        # Get embedding vectors
	    emb_vec = logits['emb_vec']
        # Calculate loss and normalisation factor
        loss,norm = ops.deepattractornet_sigmoid_loss(partioning, spectrogram_targets, mix_to_mask, energybins, emb_vec,
                            seq_length,self.batch_size)

        return loss,norm
