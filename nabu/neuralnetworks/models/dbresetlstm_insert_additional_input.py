"""@file dbresetlstm_insert_additional_input.py
contains de DBResetLSTM class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import numpy as np
import concat


class DBResetLSTM(model.Model):
	"""A deep bidirectional reset LSTM classifier. Identical to DBResetLSTM form dbresetlstm.py. But an additional input
	 can be added after one of the layers"""

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

		# the bresetlstm layer
		num_units = int(self.conf['num_units'])
		num_lay = int(self.conf['num_layers'])
		t_resets = self.conf['t_reset']
		t_resets = t_resets.split(' ')
		if len(t_resets) == 1:
			t_resets = t_resets * num_lay
		t_resets = map(int, t_resets)
		if any([t_resets[l+1] < t_resets[l] for l in range(num_lay-1)]):
			raise ValueError('T_reset in next layer must be equal to or bigger than T_reset in current layer')
		if 'group_size' in self.conf:
			group_sizes = self.conf['group_size']
			group_sizes = group_sizes.split(' ')
		else:
			group_sizes = '1'
		if len(group_sizes) == 1:
			group_sizes = group_sizes * num_lay
		group_sizes = map(int, group_sizes)
		if any([np.mod(t_res, group_size) != 0 for t_res, group_size in zip(t_resets, group_sizes)]):
			raise ValueError('t_reset should be a multiple of group_size')

		if 'forward_reset' in self.conf:
			forward_reset = self.conf['forward_reset'] == 'True'
		else:
			forward_reset = True
		if 'backward_reset' in self.conf:
			backward_reset = self.conf['backward_reset'] == 'True'
		else:
			backward_reset = True

		if 'separate_directions' in self.conf:
			separate_directions = self.conf['separate_directions'] == 'True'
		else:
			separate_directions = False

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
	
		# code not available for multiple inputs!!
		if len(inputs) > 2:
			raise Exception('The implementation of DBLSTM expects 2 input and not %d' % len(inputs))
		else:
			add_input = inputs[1]
			inputs = inputs[0]

		# combine output of first lstm layers with additional input. combine_layer indicates which layer will receive
		# the combination as input
		self.combine_layer = int(self.conf['combine_layer']) - 1

		# concat model to combine output of first  combine_layer lstm layers with additional input
		self.concat_model = concat.Concat(self.conf)

		with tf.variable_scope(self.scope):
			if is_training and float(self.conf['input_noise']) > 0:
				inputs = inputs + tf.random_normal(
					tf.shape(inputs),
					stddev=float(self.conf['input_noise']))

			logits = inputs

			for l in range(num_lay):

				blstm = layer.BResetLSTMLayer(
					num_units=num_units,
					t_reset=t_resets[l],
					group_size=group_sizes[l],
					forward_reset=forward_reset,
					backward_reset=backward_reset,
					layer_norm=layer_norm,
					recurrent_dropout=recurrent_dropout,
					activation_fn=activation_fn)

				if l == 0:
					# expand the dimension of inputs since the reset lstm expect multistate input
					multistate_input = tf.expand_dims(logits, 2)
					num_replicates = int(float(t_resets[l]) / float(group_sizes[l]))
					multistate_input = tf.tile(multistate_input, tf.constant([1, 1, num_replicates, 1]))
					if forward_reset:
						for_forward = multistate_input
					else:
						for_forward = logits
					if backward_reset:
						for_backward = multistate_input
					else:
						for_backward = logits
				else:
					for_forward, for_backward = permute_versions(
						logits_multistate, logits, input_seq_length, t_resets[l], t_resets[l-1], group_sizes[l],
						group_sizes[l-1], forward_reset, backward_reset, separate_directions)

				if l == self.combine_layer:
					multiplicates = np.ones(len(for_forward.get_shape()), np.int).tolist()
					multiplicates[1] = tf.shape(for_forward)[1]
					multiplicates[2] = tf.shape(for_forward)[2]
					multiplicates = tf.stack(multiplicates)

					add_input_broadcast = tf.expand_dims(tf.expand_dims(add_input, 1), -2)
					add_input_broadcast = tf.tile(add_input_broadcast, multiplicates)

					for_forward = tf.concat([for_forward, add_input_broadcast], -1)
					for_backward = tf.concat([for_backward, add_input_broadcast], -1)

				logits, logits_multistate = blstm(for_forward, for_backward, input_seq_length, 'layer' + str(l))

			if is_training and float(self.conf['dropout']) < 1:
				raise Exception('dropout not yet implemented for state reset lstm')
				# logits = tf.nn.dropout(logits, float(self.conf['dropout']))

			output = tf.concat(logits, -1)

		return output


def permute_versions(
		replicas, actual_outputs, sequence_length, t_reset, previous_t_reset, group_size, previous_group_size,
		forward_reset=True,	backward_reset=True, separate_directions=False):
	forward_output = actual_outputs[0]
	backward_output = actual_outputs[1]
	forward_replicas = replicas[0]
	backward_replicas = replicas[1]

	batch_size = forward_output.get_shape()[0]
	max_length = tf.shape(forward_output)[1]

	int_dtype = sequence_length.dtype
	if np.mod(t_reset, group_size) != 0 or np.mod(previous_t_reset, previous_group_size) != 0:
		raise ValueError('Reset period must be multiple of group size')
	num_replicas = int(float(t_reset)/float(group_size))
	previous_num_replicas = int(float(previous_t_reset)/float(previous_group_size))
	# the output replicas need to be permuted correctly such that the next layer receives
	# the replicas in the correct order

	# T: [B,1, 1]
	T = tf.expand_dims(tf.expand_dims(sequence_length, -1), -1)

	# numbers_to_maxT: [B,Tmax,k]
	numbers_to_maxT = tf.range(0, max_length)
	numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, 0), -1)
	numbers_to_maxT = tf.tile(numbers_to_maxT, [batch_size, 1, num_replicas])
	reversed_numbers_to_maxT = T - 1 - numbers_to_maxT

	# numbers_to_k: [B,Tmax,k]
	numbers_to_k = tf.expand_dims(tf.expand_dims(range(0, num_replicas), 0), 0)
	numbers_to_k = tf.tile(numbers_to_k, [batch_size, max_length, 1])

	# taus
	max_tau = previous_t_reset - 1
	max_tau = np.expand_dims(np.expand_dims(np.expand_dims(max_tau, -1), -1), -1)
	max_tau_tf = tf.tile(tf.constant(max_tau, dtype=int_dtype), [batch_size, max_length, num_replicas])
	tau_forward = tf.mod(numbers_to_maxT - group_size * numbers_to_k, t_reset)
	tau_forward = tf.minimum(tau_forward, max_tau_tf)
	tau_backward = tf.mod(reversed_numbers_to_maxT - group_size * numbers_to_k, t_reset)
	tau_backward = tf.minimum(tau_backward, max_tau_tf)

	# forward for forward
	forward_indices_for_forward = tf.cast(tf.mod(
		tf.ceil(tf.truediv(numbers_to_maxT - tau_forward, previous_group_size)), previous_num_replicas), int_dtype)

	# backward for forward
	backward_indices_for_forward = tf.cast(tf.mod(
		tf.ceil(tf.truediv(reversed_numbers_to_maxT - tau_forward, previous_group_size)), previous_num_replicas), int_dtype)

	# backward for backward
	backward_indices_for_backward = tf.cast(tf.mod(
		tf.ceil(tf.truediv(reversed_numbers_to_maxT - tau_backward, previous_group_size)), previous_num_replicas), int_dtype)

	# forward for backward
	forward_indices_for_backward = tf.cast(tf.mod(
		tf.ceil(tf.truediv(numbers_to_maxT - tau_backward, previous_group_size)), previous_num_replicas), int_dtype)

	# ra1: [B,Tmax,k]
	ra1 = tf.range(batch_size)
	ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
	ra1 = tf.tile(ra1, [1, max_length, num_replicas])
	ra2 = tf.range(max_length)
	ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
	ra2 = tf.tile(ra2, [batch_size, 1, num_replicas])
	stacked_forward_indices_for_forward = tf.stack([ra1, ra2, forward_indices_for_forward], axis=-1)
	stacked_backward_indices_for_forward = tf.stack([ra1, ra2, backward_indices_for_forward], axis=-1)
	stacked_forward_indices_for_backward = tf.stack([ra1, ra2, forward_indices_for_backward], axis=-1)
	stacked_backward_indices_for_backward = tf.stack([ra1, ra2, backward_indices_for_backward], axis=-1)

	if forward_reset:
		forward_for_forward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_forward)
	else:
		forward_for_forward = forward_output

	if backward_reset:
		backward_for_backward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_backward)
	else:
		backward_for_backward = backward_output

	if forward_reset and backward_reset:
		backward_for_forward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_forward)
		forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward)
	elif forward_reset and not backward_reset:
		backward_for_forward = tf.tile(tf.expand_dims(backward_output, -2), [1, 1, num_replicas, 1])
		forward_for_backward = forward_output
	elif not forward_reset and backward_reset:
		backward_for_forward = backward_output
		forward_for_backward = tf.tile(tf.expand_dims(forward_output, -2), [1, 1, num_replicas, 1])

	if separate_directions:
		for_forward = forward_for_forward
		for_backward = backward_for_backward
	else:
		for_forward = tf.concat((forward_for_forward, backward_for_forward), -1)
		for_backward = tf.concat((forward_for_backward, backward_for_backward), -1)

	return for_forward, for_backward

# def permute_versions(replicas, sequence_length, t_reset, previous_t_reset, group_size, previous_group_size):
# 	forward_replicas = replicas[0]
# 	backward_replicas = replicas[1]
#
# 	batch_size = forward_replicas.get_shape()[0]
# 	max_length = tf.shape(forward_replicas)[1]
# 	int_dtype = sequence_length.dtype
# 	if np.mod(t_reset, group_size) != 0 or np.mod(previous_t_reset, previous_group_size) != 0:
# 		raise ValueError('Reset period must be multiple of group size')
# 	num_replicas = int(float(t_reset)/float(group_size))
# 	previous_num_replicas = int(float(previous_t_reset)/float(previous_group_size))
# 	# the output replicas need to be permuted correctly such that the next layer receives
# 	# the replicas in the correct order
#
# 	# T: [B,1, 1]
# 	T = tf.expand_dims(tf.expand_dims(sequence_length, -1), -1)
#
# 	# numbers_to_maxT: [B,Tmax,k]
# 	numbers_to_maxT = tf.range(0, max_length)
# 	numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, 0), -1)
# 	numbers_to_maxT = tf.tile(numbers_to_maxT, [batch_size, 1, num_replicas])
# 	reversed_numbers_to_maxT = T - 1 - numbers_to_maxT
#
# 	# numbers_to_k: [B,Tmax,k]
# 	numbers_to_k = tf.expand_dims(tf.expand_dims(range(0, num_replicas), 0), 0)
# 	numbers_to_k = tf.tile(numbers_to_k, [batch_size, max_length, 1])
#
# 	# taus
# 	max_tau = previous_t_reset - 1
# 	max_tau = np.expand_dims(np.expand_dims(np.expand_dims(max_tau, -1), -1), -1)
# 	max_tau_tf = tf.tile(tf.constant(max_tau, dtype=int_dtype), [batch_size, max_length, num_replicas])
# 	tau_forward = tf.mod(numbers_to_maxT - group_size * numbers_to_k, t_reset)
# 	condition_forward = tau_forward <= max_tau
# 	tau_forward = tf.where(condition_forward, x=tau_forward, y=max_tau_tf)
# 	tau_backward = tf.mod(reversed_numbers_to_maxT - group_size * numbers_to_k, t_reset)
# 	condition_backward = tau_backward <= max_tau
# 	tau_backward = tf.where(condition_backward, x=tau_backward, y=max_tau_tf)
#
# 	# forward for forward
# 	forward_indices_for_forward = tf.cast(tf.mod(
# 		tf.ceil(tf.truediv(numbers_to_maxT - tau_forward, previous_group_size)), previous_num_replicas), int_dtype)
#
# 	# backward for forward
# 	backward_indices_for_forward = tf.cast(tf.mod(
# 		tf.ceil(tf.truediv(reversed_numbers_to_maxT - tau_forward, previous_group_size)), previous_num_replicas), int_dtype)
#
# 	# backward for backward
# 	backward_indices_for_backward = tf.cast(tf.mod(
# 		tf.ceil(tf.truediv(reversed_numbers_to_maxT - tau_backward, previous_group_size)), previous_num_replicas), int_dtype)
#
# 	# forward for backward
# 	forward_indices_for_backward = tf.cast(tf.mod(
# 		tf.ceil(tf.truediv(numbers_to_maxT - tau_backward, previous_group_size)), previous_num_replicas), int_dtype)
#
# 	# ra1: [B,Tmax,k]
# 	ra1 = tf.range(batch_size)
# 	ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
# 	ra1 = tf.tile(ra1, [1, max_length, num_replicas])
# 	ra2 = tf.range(max_length)
# 	ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
# 	ra2 = tf.tile(ra2, [batch_size, 1, num_replicas])
# 	stacked_forward_indices_for_forward = tf.stack([ra1, ra2, forward_indices_for_forward], axis=-1)
# 	forward_for_forward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_forward)
# 	stacked_backward_indices_for_forward = tf.stack([ra1, ra2, backward_indices_for_forward], axis=-1)
# 	backward_for_forward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_forward)
# 	stacked_backward_indices_for_backward = tf.stack([ra1, ra2, backward_indices_for_backward], axis=-1)
# 	backward_for_backward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_backward)
# 	stacked_forward_indices_for_backward = tf.stack([ra1, ra2, forward_indices_for_backward], axis=-1)
# 	forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward)
#
# 	replicas_for_forward = tf.concat((forward_for_forward, backward_for_forward), -1)
# 	replicas_for_backward = tf.concat((forward_for_backward, backward_for_backward), -1)
#
# 	return replicas_for_forward, replicas_for_backward
