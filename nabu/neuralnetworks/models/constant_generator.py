"""@file constant_generator.py
contains the ConstantGenerator class"""

import tensorflow as tf
import model
import numpy as np


class ConstantGenerator(model.Model):
	"""Returns a model that simply concatenates inputs"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		concatenate the inputs over the last dimension.

		Args:
			inputs: the inputs to concatenate, this is a list of
				[batch_size x time x ...] tensors and/or [batch_size x ...] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, the concatenated inputs
		"""

		constant_value = float(self.conf['constant_value'])
		tensor_shape = map(int, self.conf['tensor_shape'].split(','))
		if 'include_batch_size' in self.conf and self.conf['include_batch_size'] == 'True':
			batch_size = inputs[0].get_shape()[0]
			tensor_shape = [batch_size] + tensor_shape

		output = constant_value * np.ones(tensor_shape, dtype=np.float32)

		if is_training and 'output_noise' in self.conf:
			output_noise = map(float, self.conf['output_noise'].split(' '))
			output = output + tf.random_normal(tf.shape(output), stddev=output_noise)

		return output