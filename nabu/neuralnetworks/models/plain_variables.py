"""@file plain_variables.py
contains de PlainVariables class"""

import tensorflow as tf
import model


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
			vector_set = tf.get_variable(
				'vector_set', initializer=tf.truncated_normal(
					[int(self.conf['tot_vecs']), int(self.conf['vec_dim'])],
					stddev=tf.sqrt(2/float(self.conf['vec_dim']))))

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
			vector_set = tf.get_variable(
				'vector_set', initializer=tf.truncated_normal(
					[int(self.conf['tot_vecs']), int(self.conf['vec_dim'])],
					stddev=tf.sqrt(2/float(self.conf['vec_dim']))))

			inputs = tf.expand_dims(inputs, -1)

			output = tf.gather_nd(vector_set, inputs)
			output = tf.reshape(output,[tf.shape(output)[0],1,-1])

			return output
