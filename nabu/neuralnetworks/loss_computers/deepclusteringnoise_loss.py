"""@file deepclusteringnoise_loss.py
contains the DeepclusteringnoiseLoss"""

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops


class DeepclusteringnoiseLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss"""

    def __call__(self, targets, logits, seq_length):
        """
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
        """

        binary_target = targets['binary_targets']  # partition targets
        energybins = targets['usedbins']  # which bins contain enough energy
        seq_length = seq_length['bin_emb']  # Sequence length of utterances in batch
        emb_vec = logits['bin_emb']  # Embeddingvectors
        alpha = logits['noise_filter']  # alpha outputs
        noisybins = targets['noisybins']  # Dominates noise the cell
        ideal_ratio = targets['rel_speech_targets']  # Ideal ratio masker to filter noise
        # Calculate cost and normalisation
        loss, norm = ops.deepclustering_noise_loss(
            binary_target, noisybins, ideal_ratio, emb_vec, alpha, energybins, seq_length, self.batch_size)

        return loss, norm


class DeepclusteringnoiseRatAsWeightLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss"""

    def __call__(self, targets, logits, seq_length):
        """
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
        """

        binary_target = targets['binary_targets']  # partition targets
        energybins = targets['usedbins']  # which bins contain enough energy
        seq_length = seq_length['bin_emb']  # Sequence length of utterances in batch
        emb_vec = logits['bin_emb']  # Embeddingvectors
        alpha = logits['noise_filter']  # alpha outputs
        noisybins = targets['noisybins']  # Dominates noise the cell
        ideal_ratio = targets['rel_speech_targets']  # Ideal ratio masker to filter noise
        # Calculate cost and normalisation
        loss, norm = ops.deepclustering_noise_loss(
            binary_target, noisybins, ideal_ratio, emb_vec, alpha, energybins, seq_length, self.batch_size,
            rat_as_weight=True)

        return loss, norm


class DeepclusteringnoiseAlphaAsWeightLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss"""

    def __call__(self, targets, logits, seq_length):
        """
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
        """

        binary_target = targets['binary_targets']  # partition targets
        energybins = targets['usedbins']  # which bins contain enough energy
        seq_length = seq_length['bin_emb']  # Sequence length of utterances in batch
        emb_vec = logits['bin_emb']  # Embeddingvectors
        alpha = logits['noise_filter']  # alpha outputs
        noisybins = targets['noisybins']  # Dominates noise the cell
        ideal_ratio = targets['rel_speech_targets']  # Ideal ratio masker to filter noise
        # Calculate cost and normalisation
        loss, norm = ops.deepclustering_noise_loss(
            binary_target, noisybins, ideal_ratio, emb_vec, alpha, energybins, seq_length, self.batch_size,
            rat_as_weight=False, alpha_as_weight=True)

        return loss, norm


class DeepclusteringnoiseSnrTargetLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss. See 'IDEAL RATIO MASK ESTIMATION USING DEEP NEURAL NETWORKS FOR ROBUST
    SPEECH RECOGNITION' by Narayanan et al."""
    def __init__(self, batch_size):
        self.alpha = 0.2
        self.beta = -6.0

        super(DeepclusteringnoiseSnrTargetLoss, self).__init__(batch_size)

    def __call__(self, targets, logits, seq_length):
        """
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
        """

        binary_target = targets['binary_targets']  # partition targets
        energybins = targets['usedbins']  # which bins contain enough energy
        seq_length = seq_length['bin_emb']  # Sequence length of utterances in batch
        emb_vec = logits['bin_emb']  # Embeddingvectors
        alpha = logits['noise_filter']  # alpha outputs
        noisybins = targets['noisybins']  # Dominates noise the cell
        snr = targets['snr_targets']  # Ideal ratio masker to filter noise
        desired_targets = 1/(1+tf.exp(-self.alpha*(snr - self.beta)))
        # Calculate cost and normalisation
        loss, norm = ops.deepclustering_noise_loss(
            binary_target, noisybins, desired_targets, emb_vec, alpha, energybins, seq_length, self.batch_size)

        return loss, norm


class DeepclusteringnoiseDConlyLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss"""

    def __call__(self, targets, logits, seq_length):
        """
        Compute the loss

        Ignore this class, just for analysis

        Args:
            targets: a dictionary of [batch_size x time x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x time x ...] tensors containing the logits
        seq_length: a dictionary of [batch_size] vectors containing
                the sequence lengths

        Returns:
            loss: a scalar value containing the loss (of mini-batch)
            norm: a scalar value indicating how to normalize the loss
        """

        binary_target = targets['binary_targets']  # partition targets
        energybins = targets['usedbins']  # which bins contain enough energy
        seq_length = seq_length['bin_emb']  # Sequence length of utterances in batch
        emb_vec = logits['bin_emb']  # Embeddingvectors
        alpha = logits['noise_filter']  # alpha outputs
        noisybins = targets['noisybins']  # Dominates noise the cell
        ideal_ratio = targets['rel_speech_targets']  # Ideal ratio masker to filter noise
        # Calculate cost and normalisation
        loss, norm = ops.deepclustering_noise_DC_only_loss(
            binary_target, noisybins, ideal_ratio, emb_vec, alpha, energybins, seq_length, self.batch_size)

        return loss, norm


class DeepclusteringnoiseNoiseonlyLoss(loss_computer.LossComputer):
    """A loss computer that calculates the loss"""

    def __call__(self, targets, logits, seq_length):
        """
        Compute the loss

        Ignore this class, just for analysis

        Args:
            targets: a dictionary of [batch_size x time x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x time x ...] tensors containing the logits
        seq_length: a dictionary of [batch_size] vectors containing
                the sequence lengths

        Returns:
            loss: a scalar value containing the loss (of mini-batch)
            norm: a scalar value indicating how to normalize the loss
        """

        energybins = targets['usedbins']  # which bins contain enough energy
        alpha = logits['noise_filter']  # alpha outputs
        ideal_ratio = targets['rel_speech_targets']  # Ideal ratio masker to filter noise
        # Calculate cost and normalisation
        loss, norm = ops.noise_mask_loss(ideal_ratio, alpha, energybins)

        return loss, norm