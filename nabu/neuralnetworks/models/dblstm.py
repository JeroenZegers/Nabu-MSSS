"""@file dblstm.py
contains de DBLSTM class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import math


class DBLSTM(model.Model):
	"""A deep bidirectional LSTM classifier"""

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

		# the blstm layer
		num_layers = int(self.conf['num_layers'])
		num_units_first_layer = int(self.conf['num_units'])
		if 'fac_per_layer' in self.conf:
			fac_per_layer = float(self.conf['fac_per_layer'])
		else:
			fac_per_layer = 1.0
		num_units = [
			int(math.ceil(num_units_first_layer*(fac_per_layer**l)))
			for l in range(num_layers)]

		layer_norm = self.conf['layer_norm'] == 'True'
		recurrent_dropout = float(self.conf['recurrent_dropout'])
		if 'activation_fn' in self.conf:
			if self.conf['activation_fn'] == 'tanh':
				activation_fn = tf.nn.tanh
			elif self.conf['activation_fn'] == 'relu':
				activation_fn = tf.nn.relu
			elif self.conf['activation_fn'] == 'sigmoid':
				activation_fn = tf.nn.sigmoid
			else:
				raise Exception('Undefined activation function: %s' % self.conf['activation_fn'])
		else:
			activation_fn = tf.nn.tanh

		separate_directions = False
		if 'separate_directions' in self.conf and self.conf['separate_directions'] == 'True':
			separate_directions = True

		blstm_layers = []
		for l in range(num_layers):
			blstm_layers.append(layer.BLSTMLayer(
				num_units=num_units[l],
				layer_norm=layer_norm,
				recurrent_dropout=recurrent_dropout,
				activation_fn=activation_fn,
				separate_directions=separate_directions,
				fast_version=False))
	
		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of DBLSTM expects 1 input and not %d' % len(inputs)
		else:
			inputs = inputs[0]
		if num_layers == 0:
			output = inputs
			return output

		with tf.variable_scope(self.scope):
			if is_training and float(self.conf['input_noise']) > 0:
				inputs = inputs + tf.random_normal(
					tf.shape(inputs),
					stddev=float(self.conf['input_noise']))

			logits = inputs

			if separate_directions:
				logits = (logits, logits)

			for l in range(num_layers):
				logits = blstm_layers[l](logits, input_seq_length, 'layer' + str(l))

				if is_training and float(self.conf['dropout']) < 1:
					logits = tf.nn.dropout(logits, float(self.conf['dropout']))

			output = logits
			if separate_directions:
				output = tf.concat(output, 2)

		return output
