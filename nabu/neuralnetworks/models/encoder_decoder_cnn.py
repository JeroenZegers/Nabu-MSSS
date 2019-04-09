"""@file encoder_decoder_cnn.py
contains de EncoderDecoderCNN class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import numpy as np
import copy
import math


class EncoderDecoderCNN(model.Model):
	"""A CNN classifier with encoder-decoder shape
	(https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)
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
			kernel_size_lay1 = map(int, self.conf['filters'].split(' '))
		elif 'filter_size_t' in self.conf and 'filter_size_f' in self.conf:
			kernel_size_t_lay1 = int(self.conf['filter_size_t'])
			kernel_size_f_lay1 = int(self.conf['filter_size_f'])
			kernel_size_lay1 = [kernel_size_t_lay1, kernel_size_f_lay1]
		else:
			raise ValueError('Kernel convolution size not specified.')
		if 'filter_size_t' in self.conf and 'filter_size_f' in self.conf:
			kernel_size_t_fac_after_pool = float(self.conf['filter_size_t_fac_after_pool'])
			kernel_size_f_fac_after_pool = float(self.conf['filter_size_f_fac_after_pool'])
			kernel_fac_after_pool = [kernel_size_t_fac_after_pool, kernel_size_f_fac_after_pool]
		else:
			kernel_fac_after_pool = [1, 1]

		f_pool_rate = int(self.conf['f_pool_rate'])
		t_pool_rate = int(self.conf['t_pool_rate'])
		num_encoder_layers = int(self.conf['num_encoder_layers'])
		num_decoder_layers = num_encoder_layers
		num_centre_layers = int(self.conf['num_centre_layers'])
		num_filters_1st_layer = int(self.conf['num_filters_1st_layer'])
		fac_per_layer = float(self.conf['fac_per_layer'])
		num_filters_enc = [
			int(math.ceil(num_filters_1st_layer*(fac_per_layer**l)))
			for l in range(num_encoder_layers)]
		num_filters_dec = num_filters_enc[::-1]
		num_filters_dec = num_filters_dec[1:] + [(int(self.conf['num_output_filters']))]

		kernel_size_enc = []
		ideal_kernel_size_enc = [kernel_size_lay1]

		bypass = self.conf['bypass']

		layer_norm = self.conf['layer_norm'] == 'True'

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

		# the encoder layers
		encoder_layers = []
		for l in range(num_encoder_layers):
			kernel_size_l = copy.deepcopy(ideal_kernel_size_enc[l])
			kernel_size_l_plus_1 = kernel_size_l
			kernel_size_l = [int(math.ceil(k)) for k in kernel_size_l]
			kernel_size_enc.append(kernel_size_l)

			num_filters_l = num_filters_enc[l]

			max_pool_filter = [1, 1]
			if np.mod(l+1, t_pool_rate) == 0:
				max_pool_filter[0] = 2
				kernel_size_l_plus_1[0] = kernel_size_l_plus_1[0] * kernel_fac_after_pool[0]
			if np.mod(l+1, f_pool_rate) == 0:
				max_pool_filter[1] = 2
				kernel_size_l_plus_1[1] = kernel_size_l_plus_1[1] * kernel_fac_after_pool[1]
			ideal_kernel_size_enc.append(kernel_size_l_plus_1)

			encoder_layers.append(layer.Conv2D(
				num_filters=num_filters_l,
				kernel_size=kernel_size_l,
				strides=(1, 1),
				padding='same',
				activation_fn=activation_fn,
				layer_norm=layer_norm,
				max_pool_filter=max_pool_filter))

		# the centre layers
		centre_layers = []
		for l in range(num_centre_layers):
			num_filters_l = num_filters_enc[-1]
			kernel_size_l = ideal_kernel_size_enc[-1]
			kernel_size_l = map(int(math.ceil()), kernel_size_l)

			centre_layers.append(layer.Conv2D(
				num_filters=num_filters_l,
				kernel_size=kernel_size_l,
				strides=(1, 1),
				padding='same',
				activation_fn=activation_fn,
				layer_norm=layer_norm,
				max_pool_filter=(1, 1)))

		# the decoder layers
		decoder_layers = []
		for l in range(num_decoder_layers):
			corresponding_encoder_l = num_encoder_layers-1-l
			num_filters_l = num_filters_dec[l]
			kernel_size_l = kernel_size_enc[corresponding_encoder_l]
			if bypass == 'unpool':
				strides = [1, 1]
			else:
				strides = encoder_layers[corresponding_encoder_l].max_pool_filter

			decoder_layers.append(layer.Conv2D(
				num_filters=num_filters_l,
				kernel_size=kernel_size_l,
				strides=strides,
				padding='same',
				activation_fn=activation_fn,
				layer_norm=layer_norm,
				max_pool_filter=(1, 1),
				transpose=True))

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of DCNN expects 1 input and not %d' % len(inputs)
		else:
			inputs = inputs[0]
		if (num_encoder_layers + num_centre_layers + num_decoder_layers) == 0:
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

			with tf.variable_scope('encoder'):
				encoder_outputs = []
				encoder_outputs_before_pool = []
				for l in range(num_encoder_layers):
					with tf.variable_scope('layer_%s' % l):

						logits, outputs_before_pool = encoder_layers[l](logits)

						encoder_outputs.append(logits)
						encoder_outputs_before_pool.append(outputs_before_pool)

						if is_training and float(self.conf['dropout']) < 1:
							raise Exception('have to check whether dropout is implemented correctly')
							# logits = tf.nn.dropout(logits, float(self.conf['dropout']))

			with tf.variable_scope('centre'):
				for l in range(num_centre_layers):
					with tf.variable_scope('layer_%s' % l):

						logits, _ = centre_layers[l](logits)

						if is_training and float(self.conf['dropout']) < 1:
							raise Exception('have to check whether dropout is implemented correctly')
							# logits = tf.nn.dropout(logits, float(self.conf['dropout']))

			with tf.variable_scope('decoder'):
				for l in range(num_decoder_layers):
					with tf.variable_scope('layer_%s' % l):
						corresponding_encoder_l = num_encoder_layers-1-l
						corresponding_encoder_output = encoder_outputs[corresponding_encoder_l]
						corresponding_encoder_output_before_pool = encoder_outputs_before_pool[corresponding_encoder_l]
						corresponding_encoder_max_pool_filter = encoder_layers[corresponding_encoder_l].max_pool_filter
						if bypass == 'True' and (num_centre_layers > 0 or l > 0):
							# don't use bypass for layer 0 if no centre layers
							decoder_input = tf.concat([logits, corresponding_encoder_output], -1)
						else:
							decoder_input = logits

						if bypass == 'unpool' and corresponding_encoder_max_pool_filter != [1, 1]:
							decoder_input = layer.unpool(
								pool_input=corresponding_encoder_output_before_pool,
								pool_output=corresponding_encoder_output, unpool_input=decoder_input,
								pool_kernel_size=corresponding_encoder_max_pool_filter,
								pool_stride=corresponding_encoder_max_pool_filter, padding='VALID')

						logits, _ = decoder_layers[l](decoder_input)

						if is_training and float(self.conf['dropout']) < 1:
							raise Exception('have to check whether dropout is implemented correctly')
							# logits = tf.nn.dropout(logits, float(self.conf['dropout']))

						# get wanted output size
						if corresponding_encoder_l == 0:
							wanted_size = tf.shape(inputs)
						else:
							wanted_size = tf.shape(encoder_outputs[corresponding_encoder_l-1])
						wanted_t_size = wanted_size[1]
						wanted_f_size = wanted_size[2]

						# get actual output size
						output_size = tf.shape(logits)
						output_t_size = output_size[1]
						output_f_size = output_size[2]

						# compensate for potential mismatch, by adding duplicates
						missing_t_size = wanted_t_size-output_t_size
						missing_f_size = wanted_f_size-output_f_size

						last_t_slice = tf.expand_dims(logits[:, -1, :, :], 1)
						duplicate_logits = tf.tile(last_t_slice, [1, missing_t_size, 1, 1])
						logits = tf.concat([logits, duplicate_logits], 1)
						last_f_slice = tf.expand_dims(logits[:, :, -1, :], 2)
						duplicate_logits = tf.tile(last_f_slice, [1, 1, missing_f_size, 1])
						logits = tf.concat([logits, duplicate_logits], 2)

			# set the shape of the logits as we know
			dyn_shape = logits.get_shape().as_list()
			dyn_shape[-2] = inputs.get_shape()[-2]
			logits.set_shape(dyn_shape)
			output = logits

		return output
