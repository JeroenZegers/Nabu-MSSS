"""@file averager.py
contains the Averager class"""

import tensorflow as tf
import model
import numpy as np
import warnings

class Averager(model.Model):
	"""Returns a model that simply averages the input"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		averages the input over the last dimension.

		Args:
			inputs: the inputs to concatenate, this is a list of
				[batch_size x time x ...] tensors and/or [batch_size x ...] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, the averaged input
		"""

		if 'einsum' in self.conf:
			einsum = self.conf['einsum']
			raise NotImplementedError(
				'I think there is an error in the implementation. It sums instead of averaing. Might be ok if weights '
				'are normalized ')
		else:
			einsum = False
			if 'average_dim' in self.conf:
				average_dim = int(self.conf['average_dim'])
			else:
				average_dim = -1

		# code not available for multiple inputs!!
		if len(inputs) > 2:
			raise 'The implementation of Averager expects 1 or 2 inputs and not %d' % len(inputs)
		else:
			input = inputs[0]
			if len(inputs) == 1:
				weights = tf.ones(tf.shape(input))
			else:
				weights = inputs[1]
				warnings.warn('The use of averager with weights has changed a bit since 2020/04/07', Warning)

		if not einsum:
			output = tf.reduce_sum(input*weights, average_dim)
			output = output / (tf.reduce_sum(weights, average_dim) + 1e-12)
		else:
			output = tf.einsum(einsum, input, weights)

		if 'activation_func' in self.conf and self.conf['activation_func'] != 'None':
			if self.conf['activation_func'] == 'tanh':
				output = tf.tanh(output)
			elif self.conf['activation_func'] == 'sigmoid':
				output = tf.sigmoid(output)
			elif self.conf['activation_func'] == 'relu':
				output = tf.nn.relu(output)
			elif self.conf['activation_func'] == 'softmax':
				output = tf.nn.softmax(output, -1)
			else:
				raise 'Activation function %s not found' % self.conf['activation_func']

		return output