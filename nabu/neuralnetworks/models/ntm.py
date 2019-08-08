"""@file ntm.py
contains de NTM class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components.rnn_cell import NTMCell
from tensorflow.python.ops.rnn import dynamic_rnn


class NTM(model.Model):
	"""A Neural Tuning Machine"""

	def __init__(self, conf, name=None):
		super(NTM, self).__init__(conf, name=name)
		if 'return_read_heads' in self.conf and self.conf['return_read_heads'] == 'True':
			self.return_read_heads = True
			self.num_outputs = 2
		else:
			self.return_read_heads = False
			self.num_outputs = 1

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
			raise 'The implementation of NTM expects 1 input and not %d' % len(inputs)
		else:
			inputs = inputs[0]

		input_size = int(inputs.get_shape()[-1])
		memory_size = int(self.conf['memory_size'])
		memory_vector_dim = int(self.conf['memory_vector_dim'])
		read_head_num = int(self.conf['read_head_num'])
		write_head_num = int(self.conf['write_head_num'])
		shift_win_size = int(self.conf['shift_win_size'])
		addressing_mode = self.conf['addressing_mode']
		if 'init_mode' in self.conf:
			init_mode = self.conf['init_mode']
		else:
			init_mode = 'constant'
		if 'init_mode_other_params' in self.conf:
			init_mode_other_params = self.conf['init_mode_other_params']
		else:
			init_mode_other_params = 'constant'

		with tf.variable_scope(self.scope):
			if is_training and float(self.conf['input_noise']) > 0:
				inputs = inputs + tf.random_normal(
					tf.shape(inputs),
					stddev=float(self.conf['input_noise']))

			ntm_cell = NTMCell(
				input_size=input_size, memory_size=memory_size, memory_vector_dim=memory_vector_dim,
				read_head_num=read_head_num, write_head_num=write_head_num, addressing_mode=addressing_mode,
				shift_win_size=shift_win_size, clip_value=20, init_mode=init_mode,
				init_mode_other_params=init_mode_other_params, reuse=tf.get_variable_scope().reuse)

			# do the forward computation
			outputs, _ = dynamic_rnn(ntm_cell, inputs, dtype=tf.float32, sequence_length=input_seq_length)

			# if read heads are not requested, only keep the first part of outputs
			if not self.return_read_heads:
				outputs = outputs[0]

		return outputs
