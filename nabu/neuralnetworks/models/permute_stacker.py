"""@file permute_stacker.py
contains the PermuteStacker class"""

import tensorflow as tf
import model
import numpy as np
import itertools


class PermuteStacker(model.Model):
	"""Returns a model that permutes along a given dimension and stacks along a different dimension"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		permutes and stacks the inputs

		Args:
			inputs: the inputs to concatenate, this is a list of
				[batch_size x time x ...] tensors and/or [batch_size x ...] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, the reshaped input
		"""

		permute_dim = int(self.conf['permute_dim'])
		stack_dim = int(self.conf['stack_dim'])

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of PermuteStacker expects 1 input and not %d' % len(inputs)
		else:
			input = inputs[0]

		with tf.variable_scope(self.scope):
			permute_dim_size = input.get_shape()[permute_dim]
			permutations = list(itertools.permutations(range(permute_dim_size), permute_dim_size))

			all_inp_perms = [tf.gather(input, perm, axis=permute_dim) for perm in permutations]
			output = tf.concat(all_inp_perms, axis=stack_dim)

		return output
