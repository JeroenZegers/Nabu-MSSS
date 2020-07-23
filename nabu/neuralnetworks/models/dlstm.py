"""@file dlstm.py
contains de DLSTM class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import math


class DLSTM(model.Model):
	"""A deep  LSTM classifier"""

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

		# the lstm layer
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

		if 'only_last_frame' in self.conf:
			only_last_frame = self.conf['only_last_frame'] == 'True'
		else:
			only_last_frame = False

		lstm_layers = []
		for l in range(num_layers):
			lstm_layers.append(layer.LSTMLayer(
				num_units=num_units[l],
				layer_norm=layer_norm,
				recurrent_dropout=recurrent_dropout,
				activation_fn=activation_fn))

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of DLSTM expects 1 input and not %d' % len(inputs)
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

			for l in range(num_layers):
				logits = lstm_layers[l](logits, input_seq_length, 'layer' + str(l))

				if is_training and float(self.conf['dropout']) < 1:
					logits = tf.nn.dropout(logits, float(self.conf['dropout']))

			output = logits

			if only_last_frame:
				if len(output.get_shape()) != 3:
					raise BaseException('Not yet implemented for rank different from 3')
				output = output[:, -1, :]

		return output
