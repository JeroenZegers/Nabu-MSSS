"""@file deepattractornet_noisefilter_loss.py
contains the DeepattractornetnoisefilterLoss"""

import loss_computer
from nabu.neuralnetworks.components import ops


class DeepattractornetnoisefilterLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss"""

    def __call__(self, targets, logits, seq_length):
        """
        Compute the loss

        Creates the operation to compute the deep attractor network loss for the connected network

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

        # To which class belongs bin
        partitioning = targets['partitioning']
        # Clean spectograms of sources
        spectrogram_targets=targets['spectrogram_targets']
        # Spectogram of the original mixture, used to mask for scoring
        mix_to_mask = targets['mix_to_mask']
        # Which bins contain enough energy
        seq_length = seq_length['emb_vec']
        emb_vec = logits['emb_vec']
        alpha = logits['alpha']

        loss, norm = ops.deepattractornet_noisefilter_loss(
            partitioning, spectrogram_targets, mix_to_mask, emb_vec, alpha, seq_length, self.batch_size)

        return loss, norm
