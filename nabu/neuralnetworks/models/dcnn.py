"""@file dcnn.py
contains de DCNN class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import numpy as np
import copy
import math


class DCNN(model.Model):
	"""A DCNN
	"""

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

		if 'filters' in self.conf:
			kernel_size = map(int, self.conf['filters'].split(' '))
		elif 'filter_size_t' in self.conf and 'filter_size_f' in self.conf:
			kernel_size_t = int(self.conf['filter_size_t'])
			kernel_size_f = int(self.conf['filter_size_f'])
			kernel_size = (kernel_size_t, kernel_size_f)
		else:
			raise ValueError('Kernel convolution size not specified.')

		f_stride = int(self.conf['f_stride'])
		t_stride = int(self.conf['t_stride'])
		num_layers = int(self.conf['num_layers'])
		num_filters_1st_layer = int(self.conf['num_filters_1st_layer'])
		if 'fac_per_layer' in self.conf:
			fac_per_layer = float(self.conf['fac_per_layer'])
		else:
			fac_per_layer = 1.0
		num_filters = [int(math.ceil(num_filters_1st_layer*(fac_per_layer**l))) for l in range(num_layers)]

		layer_norm = self.conf['layer_norm'] == 'True'
		flat_freq = self.conf['flat_freq'] == 'True'

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
			activation_fn = tf.nn.relu

		# the cnn layers
		cnn_layers = []
		for l in range(num_layers):
			num_filters_l = num_filters[l]

			cnn_layers.append(layer.Conv2D(
				num_filters=num_filters_l,
				kernel_size=kernel_size,
				strides=(t_stride, f_stride),
				padding='same',
				activation_fn=activation_fn,
				layer_norm=layer_norm))

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of DCNN expects 1 input and not %d' % len(inputs)
		else:
			inputs = inputs[0]
		if num_layers == 0:
			output = inputs
			return output

		# Convolutional layers expect input channels, making 1 here.
		inputs = tf.expand_dims(inputs, -1)
		with tf.variable_scope(self.scope):
			if is_training and float(self.conf['input_noise']) > 0:
				inputs = inputs + tf.random_normal(
					tf.shape(inputs),
					stddev=float(self.conf['input_noise']))

			logits = inputs

			with tf.variable_scope('cnn'):
				for l in range(num_layers):
					with tf.variable_scope('layer_%s' % l):

						logits, _ = cnn_layers[l](logits)

						if is_training and float(self.conf['dropout']) < 1:
							raise Exception('have to check whether dropout is implemented correctly')
							# logits = tf.nn.dropout(logits, float(self.conf['dropout']))

			if flat_freq:
				shapes = logits.get_shape().as_list()
				logits = tf.reshape(logits, [shapes[0], -1, shapes[2]*shapes[3]])
			output = logits
		return output
