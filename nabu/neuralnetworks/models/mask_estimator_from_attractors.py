"""@file mask_estimator_from_attractors.py
contains the MaskEstimatorFromAttractors class"""

import tensorflow as tf
import model


class MaskEstimatorFromAttractors(model.Model):
	"""MaskEstimatorFromAttractorsr"""

	def _get_outputs(self, inputs, input_seq_length, is_training):
		"""
		Create the variables and do the forward computation

		Args:
			inputs: the inputs to the neural network, this is a list of
				[batch_size x time x ...] tensors
			input_seq_length: The sequence lengths of the input utterances, this
				is a [batch_size] vector
			is_training: whether or not the network is in training mode

		Returns:
			- output, which is a [batch_size x time x ...] tensors
		"""

		normalization = self.conf['normalization'] == 'True'
		activation = self.conf['activation']

		thr = float(self.conf['thr'])
		use_binary_masks = self.conf['binary_masks'] == 'True'
		rescale_masks = self.conf['rescale_masks'] == 'True' and not use_binary_masks

		# Whether to store the output in tf_targets_format (BxTxFS)
		tf_targets_format = self.conf['tf_targets_format'] == 'True'
		# Whether to store the output in pit_loss_format (BxTxSF)
		pit_loss_format = 'pit_loss_format' in self.conf and self.conf['pit_loss_format'] == 'True' and not tf_targets_format

		if len(inputs) > 2:
			raise 'The implementation of MaskEstimatorFromAttractors expects 2 input and not %d' % len(inputs)
		else:
			bin_embs = inputs[0]
			attractors = inputs[1]

		bin_embs_shape = bin_embs.get_shape()
		attractors_shape = attractors.get_shape()
		batch_size = bin_embs_shape[0]
		output_dim = bin_embs_shape[-1]
		emb_dim = attractors_shape[-1]
		nrS = attractors_shape[1]
		feat_dim = output_dim/emb_dim

		bin_embs = tf.reshape(bin_embs, [batch_size, -1, feat_dim, emb_dim])

		if 'einsum' in self.conf:
			einsum = self.conf['einsum']
		else:
			einsum = False

		with tf.variable_scope(self.scope):
			if normalization:
				bin_embs = tf.nn.l2_normalize(bin_embs, axis=-1)
				attractors = tf.nn.l2_normalize(attractors, axis=-1)
			if not einsum:
				bin_embs = tf.expand_dims(bin_embs, 1)
				attractors = tf.expand_dims(tf.expand_dims(attractors, 2), 3)
				logits = tf.reduce_mean(bin_embs*attractors, -1)
			else:
				logits = tf.einsum(einsum, bin_embs, attractors)

			if activation == 'softmax':
				masks = tf.nn.softmax(logits, axis=1)
			elif activation in ['None', 'none', None]:
				masks = logits
			elif activation == 'sigmoid':
				masks = tf.nn.sigmoid(logits)
			else:
				raise BaseException()

			binary_masks = masks > thr
			binary_masks = tf.to_float(binary_masks)
			if use_binary_masks:
				masks = binary_masks
			else:
				masks = masks * binary_masks
				if rescale_masks:
					rescaled_masks = masks - tf.reduce_min(masks, axis=[1, 2, 3], keepdims=True)
					rescaled_masks = rescaled_masks * binary_masks
					rescaled_masks = rescaled_masks / (tf.reduce_max(rescaled_masks, axis=[1, 2, 3], keepdims=True) + 1e-12)
					rescaled_masks = rescaled_masks * binary_masks
					masks = rescaled_masks

			if tf_targets_format:
				masks = tf.transpose(masks, [0, 2, 3, 1])
				masks = tf.reshape(masks, [batch_size, -1, feat_dim*nrS])
			elif pit_loss_format:
				masks = tf.transpose(masks, [0, 2, 1, 3])
				masks = tf.reshape(masks, [batch_size, -1, nrS*feat_dim])

		return masks


