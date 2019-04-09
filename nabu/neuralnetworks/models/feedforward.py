"""@file feedfoward.py
contains the feedfoward dnn class"""

import tensorflow as tf
import model
import math


class Feedforward(model.Model):
	"""A feedfoward classifier"""

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

		num_layers = int(self.conf['num_layers'])
		num_units_first_layer = int(self.conf['num_units'])
		if 'fac_per_layer' in self.conf:
			fac_per_layer = float(self.conf['fac_per_layer'])
		else:
			fac_per_layer = 1.0
		num_units = [
			int(math.ceil(num_units_first_layer*(fac_per_layer**l)))
			for l in range(num_layers)]

		# activation function
		if 'activation_func' in self.conf:
			if self.conf['activation_func'] == 'tanh':
				activation_fn = tf.nn.tanh
			elif self.conf['activation_func'] == 'sigmoid':
				activation_fn = tf.nn.sigmoid
			elif self.conf['activation_func'] == 'relu':
				activation_fn = tf.nn.relu
			else:
				raise Exception('Undefined activation function: %s' % self.conf['activation_func'])
		else:
			activation_fn = tf.nn.tanh

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of Feedforward expects 1 input and not %d' % len(inputs)
		else:
			inputs = inputs[0]

		with tf.variable_scope(self.scope):
			if is_training and float(self.conf['input_noise']) > 0:
				inputs = inputs + tf.random_normal(
					tf.shape(inputs),
					stddev=float(self.conf['input_noise']))

			logits = inputs

			for l in range(num_layers):
				logits = tf.contrib.layers.fully_connected(
					inputs=logits,
					num_outputs=num_units[l],
					activation_fn=activation_fn)

			if is_training and float(self.conf['dropout']) < 1:
				logits = tf.nn.dropout(logits, float(self.conf['dropout']))

			output = logits

		return output
