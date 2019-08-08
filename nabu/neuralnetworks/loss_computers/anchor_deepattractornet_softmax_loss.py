"""@file anchor_deepattractornet_softmax_loss.py
contains the AnchorDeepattractornetSoftmaxLoss"""

import loss_computer
from nabu.neuralnetworks.components import ops


class AnchorDeepattractornetSoftmaxLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network with softmax loss, using anchors

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.anchor_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, seq_length, self.batch_size, activation='softmax')

		return loss, norm


class AnchorNormDeepattractornetSoftmaxLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network with softmax loss, using anchors. Embeddings will be
		 normalized.

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.anchor_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, seq_length, self.batch_size, activation='softmax',
			normalize=True)

		return loss, norm


class WeightedAnchorNormDeepattractornetSoftmaxLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network with softmax loss, using weighted anchors.
		Embeddings will be normalized.

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']
		spk_weights = logits['spk_weights']
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.weighted_anchor_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, spk_weights, seq_length, self.batch_size, activation='softmax',
			normalize=True)

		return loss, norm


class TimeAnchorDeepattractornetSoftmaxLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network with softmax loss, using anchors

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.time_anchor_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, seq_length, self.batch_size, activation='softmax')

		return loss, norm


class AnchorDeepattractornetLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network using anchors

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']

		time_anchors_flag = self.lossconf['time_anchors_flag'] == 'True'
		av_anchors_time_flag = (self.lossconf['av_anchors_time_flag'] == 'True') and time_anchors_flag
		activation = self.lossconf['activation']
		normalize_embs = self.lossconf['normalize_embs'] == 'True'
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.anchor_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, seq_length, self.batch_size, activation=activation,
			normalize=normalize_embs, time_anchors_flag=time_anchors_flag, av_anchors_time_flag=av_anchors_time_flag)

		return loss, norm


class TimeAnchorNormDeepattractornetSoftmaxLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network with softmax loss, using anchors. Embeddings will be
		 normalized.

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.time_anchor_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, seq_length, self.batch_size, activation='softmax',
			normalize=True)

		return loss, norm


class TimeAnchorReadHeadsNormDeepattractornetSoftmaxLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network with softmax loss, using anchors. Embeddings will be
		 normalized. Use read heads for assignments.

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']
		read_heads = logits['read_heads']
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.time_anchor_read_heads_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, read_heads, seq_length, self.batch_size,
			activation='softmax', normalize=True)

		return loss, norm


class TimeAnchorReadHeadsNormDeepattractornetSoftmaxFramebasedLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the deep attractor network with softmax loss, using anchors. Embeddings will be
		 normalized. Use read heads for assignments.

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
		# Clean spectograms of sources
		spectrogram_targets = targets['multi_targets']

		# Spectogram of the original mixture, used to mask for scoring
		mix_to_mask = targets['mix_to_mask']

		# Length of sequences
		seq_length = seq_length['bin_emb']
		# Logits (=output network)
		emb_vec = logits['bin_emb']
		anchors = logits['anchors']
		read_heads = logits['read_heads']
		# calculate loss and normalisation factor of mini-batch
		loss, norm = ops.time_anchor_read_heads_deepattractornet_loss(
			spectrogram_targets, mix_to_mask, emb_vec, anchors, read_heads, seq_length, self.batch_size,
			activation='softmax', normalize=True, frame_based=True)

		return loss, norm
