"""@file deepattractornet_loss.py
contains the DeepattractornetSoftmaxLoss and DeepattractornetSigmoidLoss"""

import loss_computer
from nabu.neuralnetworks.components import ops
import warnings

class DeepattractornetLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss"""

    def __call__(self, targets, logits, seq_length):
        """
        Compute the loss

        Creates the operation to compute the deep attractor loss

        Args:
            targets: a dictionary of [batch_size x time x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x time x ...] tensors containing the logits
            seq_length: a dictionary of [batch_size] vectors containing
                the sequence lengths

        Returns:
            loss: a scalar value containing the loss
            norm: a scalar value indicating how to normalize the loss
        """
        warnings.warn('Since 2020/03/02 the norm is computed in the same way as for PIT.', Warning)

        activation = self.lossconf['activation']
        if 'frame_based' in self.lossconf:
            frame_based = self.lossconf['frame_based'] in ['true', 'True']
        else:
            frame_based = False

        # To which class belongs bin
        partioning = targets['binary_targets']

        # Clean spectograms of sources
        spectrogram_targets = targets['multi_targets']

        # Spectogram of the original mixture, used to mask for scoring
        mix_to_mask = targets['mix_to_mask']

        # Which bins contain enough energy
        energybins = targets['usedbins']
        # Length of sequences
        seq_length = seq_length['bin_emb']
        # Logits (=output network)
        emb_vec = logits['bin_emb']
        # calculate loss and normalisation factor of mini-batch
        loss, norm = ops.deepattractornet_loss(
            partioning, spectrogram_targets, mix_to_mask, energybins, emb_vec, seq_length, self.batch_size,
            activation=activation, frame_based=frame_based)

        return loss, norm
