"""@file dblstm.py
contains de DBLSTM class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import math
import numpy as np
import warnings

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

		# Taking only the last frame output makes less sense in a bi-directional network
		only_last_frame = 'only_last_frame' in self.conf and self.conf['only_last_frame'] == 'True'

		separate_directions = False
		if 'separate_directions' in self.conf and self.conf['separate_directions'] == 'True':
			separate_directions = True

		allow_more_than_3dim = False
		if 'allow_more_than_3dim' in self.conf and self.conf['allow_more_than_3dim'] == 'True':
			# Assuming time dimension is one to last
			allow_more_than_3dim = True

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

			input_shape = inputs.get_shape()
			input_reshaped = False
			if len(input_shape) > 3:
				if allow_more_than_3dim:
					batch_size = input_shape[0]
					other_dims = input_shape[1:-2]
					num_inp_units = input_shape[-1]
					inputs = tf.reshape(inputs, [batch_size * np.prod(other_dims), -1, num_inp_units])
					input_seq_length = tf.expand_dims(input_seq_length, -1)
					input_seq_length = tf.tile(input_seq_length, [1, np.prod(other_dims)])
					input_seq_length = tf.reshape(input_seq_length, [-1])
					input_reshaped = True
				else:
					raise BaseException('Input has to many dimensions')

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

			if input_reshaped:
				output_shape = output.get_shape()
				num_output_units = output_shape[-1]
				output = tf.reshape(output, tf.stack([batch_size] + other_dims.as_list() + [-1] + [num_output_units], 0))

			if only_last_frame:
				output_rank = len(output.get_shape())
				if output_rank == 3:
					output = output[:, -1, :]
				elif output_rank == 4 and allow_more_than_3dim:
					output = output[:, :, -1, :]
				else:
					raise BaseException('Not yet implemented for rank different from 3 (or 4)')

		return output
