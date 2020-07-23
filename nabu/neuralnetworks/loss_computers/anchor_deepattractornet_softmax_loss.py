"""@file anchor_deepattractornet_softmax_loss.py
contains the AnchorDeepattractornetSoftmaxLoss"""

import loss_computer
from nabu.neuralnetworks.components import ops
import warnings
import tensorflow as tf


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
		warnings.warn('In following versions it will be required to use the AnchorDeepattractornetLoss', Warning)
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
		warnings.warn('In following versions it will be required to use the AnchorDeepattractornetLoss', Warning)
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
		warnings.warn('In following versions it will be required to use the AnchorDeepattractornetLoss', Warning)
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
		warnings.warn('In following versions it will be required to use the AnchorDeepattractornetLoss', Warning)
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

		if 'speaker_logits' in logits:
			# Assuming dimensions are B x T x S
			speaker_logits = logits['speaker_logits']
			av_speaker_logits_time_flag = self.lossconf['av_speaker_logits_time_flag'] == 'True'
		else:
			speaker_logits = None

		if 'anchors_scale' in logits:
			# Assuming dimensions are B x T x S
			anchors_scale = logits['anchors_scale']
			anchors_scale = anchors_scale[0, 0]
		else:
			anchors_scale = None

		time_anchors_flag = self.lossconf['time_anchors_flag'] == 'True'
		av_anchors_time_flag = (self.lossconf['av_anchors_time_flag'] == 'True') and time_anchors_flag
		activation = self.lossconf['activation']
		normalize_embs = self.lossconf['normalize_embs'] == 'True'
		normalize_anchors = self.lossconf['normalize_anchors'] == 'True'
		if 'do_square' in self.lossconf:
			do_square = self.lossconf['do_square'] == 'True'
		else:
			do_square = True

		with tf.name_scope('anchor_deepattractornet_loss'):

			feat_dim = spectrogram_targets.get_shape()[2]
			emb_dim = anchors.get_shape()[-1]
			time_dim = tf.shape(anchors)[1]
			nrS = spectrogram_targets.get_shape()[3]

			V = tf.reshape(emb_vec, [self.batch_size, -1, feat_dim, emb_dim], name='V')  # dim: (B x T x F x D)
			if normalize_embs:
				V = V / (tf.norm(V, axis=-1, keepdims=True) + 1e-12)
			time_dim = tf.shape(V)[1]

			if not time_anchors_flag:
				anchors = tf.tile(tf.expand_dims(tf.expand_dims(anchors, 0), 0), [self.batch_size, time_dim, 1, 1])  # dim: (B x T x S x D)

			if normalize_anchors:
				anchors = anchors / (tf.norm(anchors, axis=-1, keepdims=True) + 1e-12)

			if speaker_logits is not None:
				speaker_logits = tf.expand_dims(speaker_logits, -1)
				if av_speaker_logits_time_flag:
					speaker_logits = tf.reduce_mean(speaker_logits, 1, keepdims=True)
				anchors *= speaker_logits

			if anchors_scale is not None:
				anchors *= anchors_scale

			if av_anchors_time_flag:
				anchors = tf.reduce_mean(anchors, axis=1, keepdims=True)
				anchors = tf.tile(anchors, [1, time_dim, 1, 1])

			prod_1 = tf.matmul(V, anchors, transpose_a=False, transpose_b=True, name='AVT')

			if activation == 'softmax':
				masks = tf.nn.softmax(prod_1, axis=-1, name='M')  # dim: (B x T x F x nrS)
			elif activation in ['None', 'none', None]:
				masks = prod_1
			elif activation == 'sigmoid':
				masks = tf.nn.sigmoid(prod_1, name='M')
			else:
				masks = tf.nn.sigmoid(prod_1, name='M')

			X = tf.expand_dims(mix_to_mask, -1, name='X')  # dim: (B x T x F x 1)
			reconstructions = tf.multiply(masks, X)  # dim: (B x T x F x nrS)
			reconstructions = tf.transpose(reconstructions, perm=[3, 0, 1, 2])  # dim: (nrS x B x T x F)

			S = tf.transpose(spectrogram_targets, [3, 0, 1, 2])  # nrS x B x T x F

			if 'vad_targets' in targets:
				overlap_weight = float(self.lossconf['overlap_weight'])
				vad_sum = tf.reduce_sum(targets['vad_targets'], -1)
				bin_weights = tf.where(
					vad_sum > 1,
					tf.ones([self.batch_size, time_dim]) * overlap_weight,
					tf.ones([self.batch_size, time_dim]))
				bin_weights = tf.expand_dims(bin_weights, -1)  # broadcast the frame weights to all bins
				norm = tf.reduce_sum(bin_weights) * tf.to_float(feat_dim)
			else:
				bin_weights = None
				norm = tf.to_float(tf.reduce_sum(seq_length) * feat_dim)

			loss = ops.base_pit_loss(reconstructions, S, bin_weights=bin_weights, overspeakererized=False, do_square=do_square)

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
		warnings.warn('In following versions it will be required to use the AnchorDeepattractornetLoss', Warning)
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
		warnings.warn('In following versions it will be required to use the AnchorDeepattractornetLoss', Warning)
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
		warnings.warn('In following versions it will be required to use the AnchorDeepattractornetLoss', Warning)
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
