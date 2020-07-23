"""@file plain_variables.py
contains de PlainVariables class"""

import tensorflow as tf
import model
import numpy as np


class PlainVariables(model.Model):
	"""Returns vectors"""

	def _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		Create the variables

		Args:
			inputs: None
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, which is a [tot_vecs x vec_dim]
				tensor
		"""

		with tf.variable_scope(self.scope):
			# the complete vector set
			array_shape = [int(self.conf['tot_vecs']), int(self.conf['vec_dim'])]
			if 'init_value' in self.conf:
				init_value = float(self.conf['init_value'])
				initializer = tf.constant_initializer(np.ones(array_shape) * init_value)
			else:
				initializer = tf.truncated_normal(array_shape, stddev=tf.sqrt(2/float(self.conf['vec_dim'])))
				array_shape = None

			floor_val = None
			ceil_val = None
			if 'floor_val' in self.conf and self.conf['floor_val'] != 'None':
				floor_val = float(self.conf['floor_val'])
			if 'ceil_val' in self.conf and self.conf['ceil_val'] != 'None':
				ceil_val = float(self.conf['ceil_val'])
			constraint = None
			if floor_val or ceil_val:
				if not ceil_val:
					ceil_val = np.infty
				if not floor_val:
					floor_val = -np.infty
				constraint = lambda x: tf.clip_by_value(x, floor_val, ceil_val)

			# vector_set = tf.get_variable('vector_set', initializer=initializer, constraint=constraint)
			vector_set = tf.get_variable('vector_set', shape=array_shape, initializer=initializer, constraint=constraint)

		if 'normalize' in self.conf and self.conf['normalize'] == 'True':
			vector_set = vector_set/(tf.norm(vector_set, axis=-1, keepdims=True) + 1e-12)

		if 'single_scale_weight' in self.conf and self.conf['single_scale_weight'] == 'True':
			scale = tf.get_variable('single_scale', initializer=tf.constant_initializer(1.0), shape=[1])
			vector_set *= scale

		return vector_set


class PlainVariablesWithIndexing(model.Model):
	"""Returns vectors on indexing"""

	def  _get_outputs(self, inputs, input_seq_length=None, is_training=None):
		"""
		Create the variables and index them

		Args:
			inputs: the indexes, this is a list of
				[batch_size x nr_indeces] tensors
			input_seq_length: None
			is_training: None

		Returns:
			- outputs, which is a [batch_size x 1 x (nr_indeces*vec_dim)]
				tensor
		"""

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of PlainVariables expects 1 input and not %d' % len(inputs)
		else:
			inputs = inputs[0]

		with tf.variable_scope(self.scope):
			# the complete vector set
			# the complete vector set
			array_shape = [int(self.conf['tot_vecs']), int(self.conf['vec_dim'])]
			if 'init_value' in self.conf:
				init_value = float(self.conf['init_value'])
				initializer = tf.constant_initializer(np.ones(array_shape) * init_value)
			else:
				initializer = tf.truncated_normal(array_shape, stddev=tf.sqrt(2/float(self.conf['vec_dim'])))

			vector_set = tf.get_variable('vector_set', shape=array_shape, initializer=initializer)

			inputs = tf.expand_dims(inputs, -1)

			output = tf.gather_nd(vector_set, inputs)
			output = tf.reshape(output,[tf.shape(output)[0],1,-1])

			return output
