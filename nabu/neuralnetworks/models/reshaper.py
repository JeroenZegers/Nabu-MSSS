"""@file reshaper.py
contains the Reshaper class"""

import tensorflow as tf
import model
import numpy as np


class Reshaper(model.Model):
	"""Returns a model that simply reshapes the input"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		reshapes the inputs

		Args:
			inputs: the inputs to concatenate, this is a list of
				[batch_size x time x ...] tensors and/or [batch_size x ...] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, the reshaped input
		"""

		requested_shape = map(int, self.conf['requested_shape'].split(' '))
		reshape_dim = int(self.conf['reshape_dim'])

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of Reshaper expects 1 input and not %d' % len(inputs)
		else:
			input = inputs[0]

		with tf.variable_scope(self.scope):
			input_shape = tf.shape(input)
			left_in_shape = input_shape[:reshape_dim]
			right_in_shape = input_shape[reshape_dim+1:]

			reshape_dim_shape = tf.concat([left_in_shape, requested_shape, right_in_shape], 0)

			output = tf.reshape(input, reshape_dim_shape)

		return output


class DimInserter(model.Model):
	"""Returns a model that inserts a dimension"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		reshapes the inputs

		Args:
			inputs: the inputs to concatenate, this is a list of
				[batch_size x time x ...] tensors and/or [batch_size x ...] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, the reshaped input
		"""

		insert_dim = int(self.conf['insert_dim'])
		if 'duplicates' in self.conf:
			duplicates = int(self.conf['duplicates'])
		else:
			duplicates = False

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of DimInserter expects 1 input and not %d' % len(inputs)
		else:
			input = inputs[0]

		with tf.variable_scope(self.scope):
			output = tf.expand_dims(input, insert_dim)
			if duplicates:
				multiplicates = np.ones(len(output.get_shape()))
				multiplicates[insert_dim] = duplicates
				output = tf.tile(output, multiplicates)

		return output


class Transposer(model.Model):
	"""Returns a model that simply transposes the input"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		reshapes the inputs

		Args:
			inputs: the inputs to concatenate, this is a list of
				[batch_size x time x ...] tensors and/or [batch_size x ...] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, the reshaped input
		"""

		transpose_permutation = map(int, self.conf['transpose_permutation'].split(' '))

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of Transposer expects 1 input and not %d' % len(inputs)
		else:
			input = inputs[0]

		with tf.variable_scope(self.scope):
			output = tf.transpose(input, perm=transpose_permutation)

		return output