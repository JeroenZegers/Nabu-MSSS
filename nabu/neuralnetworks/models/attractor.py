"""@file attractor.py
contains the Attractor class"""

import tensorflow as tf
import model
import numpy as np


class Attractor(model.Model):
	"""Returns a model that simply calculate attractors"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		Caclulates the attractors of the first input based on the second input and third input

		Args:
			inputs: the inputs to concatenate, this is a list of
				[batch_size x time x ...] tensors and/or [batch_size x ...] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, the concatenated inputs
		"""
		num_inputs = len(inputs)

		if num_inputs == 3:
			data = inputs[0]
			labels = inputs[1]
			usedbins = inputs[2]
			data_per_spk = False
		elif num_inputs == 2:
			data = inputs[0]
			usedbins = inputs[1]
			if len(data.get_shape()) != 4:
				raise BaseException(
					'If Attractor only receives 2 inputs, data and usedbins, it expects data to have 4 dimensions '
					'(batch_size x num_speakers x time x num_un) and not %d.' % len(data.get_shape()))
			data_per_spk = True
		else:
			raise BaseException('The implementation of Attractor does not expect %d inputs' % len(inputs))

		feat_dim = usedbins.get_shape()[-1]
		output_dim = data.get_shape()[-1]
		emb_dim = output_dim/feat_dim

		if data_per_spk:
			nrS = data.get_shape()[1]
		else:
			target_dim = labels.get_shape()[-1]
			nrS = target_dim / feat_dim

		batch_size = data.get_shape()[0]

		# with tf.variable_scope(self.scope):
		usedbins = tf.to_float(usedbins)

		if data_per_spk:
			ubresh = tf.reshape(usedbins, [batch_size, 1, -1, 1], name='ebresh')
			V = tf.reshape(data, [batch_size, nrS, -1, emb_dim], name='V')
			V = tf.multiply(V, ubresh)
			Y = ubresh
			Y = tf.tile(Y, [1, nrS, 1, 1])
			Y = tf.to_float(Y)
		else:
			ubresh = tf.reshape(usedbins, [batch_size, -1, 1], name='ebresh')
			V = tf.reshape(data, [batch_size, -1, emb_dim], name='V')
			V = tf.multiply(V, ubresh)
			Y = tf.reshape(labels, [batch_size, -1, nrS], name='Y')
			Y = tf.to_float(Y)
			Y = tf.multiply(Y, ubresh)

		numerator_A = tf.matmul(Y, V, transpose_a=True, transpose_b=False, name='YTV')
		if data_per_spk:
			numerator_A = tf.squeeze(numerator_A)
			Y = tf.squeeze(Y, -1)
			# Number of bins each speaker dominates
			nb_bins_class = tf.reduce_sum(Y, axis=-1)
		else:
			# Number of bins each speaker dominates
			nb_bins_class = tf.reduce_sum(Y, axis=1)  # dim: (rank 2) (B x nrS)
		# Set number of bins of each speaker to at least 1 to avoid division by zero
		nb_bins_class = tf.where(tf.less(nb_bins_class, tf.ones_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
		# nb_bins_class = tf.where(tf.equal(nb_bins_class, tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
		nb_bins_class = tf.expand_dims(nb_bins_class, -1)  # dim: (rank 3) (B x nrS x 1)
		denominator_A = tf.tile(nb_bins_class, [1, 1, emb_dim], name='denominator_A')  # dim: (B x nrS x D)
		A = tf.divide(numerator_A, denominator_A, name='A')  # dim: (B x nrS x D)

		if 'normalization' in self.conf and self.conf['normalization'] == 'True':
			A = tf.nn.l2_normalize(A, axis=-1, epsilon=1e-12)

		return A
