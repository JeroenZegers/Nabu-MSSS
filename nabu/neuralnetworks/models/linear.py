"""@file linear.py
contains the linear class"""

import tensorflow as tf
import model
import numpy as np


class Linear(model.Model):
	"""A linear classifier"""

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
	
		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of Linear expects 1 input and not %d' % len(inputs)
		else:
			inputs = inputs[0]

		with tf.variable_scope(self.scope):
			if is_training and float(self.conf['input_noise']) > 0:
				inputs = inputs + tf.random_normal(tf.shape(inputs), stddev=float(self.conf['input_noise']))

			logits = inputs

			output_shape = map(int, self.conf['output_dims'].split(' '))
			output_size = np.prod(output_shape)

			output = tf.contrib.layers.linear(inputs=logits, num_outputs=output_size)

			if len(output_shape) > 1:
				fixed_shape = tf.shape(output)[:-1]
				output = tf.reshape(output, tf.concat([fixed_shape, tf.constant(output_shape)], -1))

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

			# dropout is not recommended
			if is_training and float(self.conf['dropout']) < 1:
				output = tf.nn.dropout(output, float(self.conf['dropout']))

			if 'last_only' in self.conf and self.conf['last_only'] == 'True':
				output = output[:, -1, :]
				output = tf.expand_dims(output, 1)

		return output
