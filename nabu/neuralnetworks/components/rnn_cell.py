"""@file rnn_cell.py
contains some customized rnn cells"""

import string

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.layers import base as base_layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import LayerRNNCell, LayerNormBasicLSTMCell, BasicRNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from nabu.neuralnetworks.components import ops
from nabu.neuralnetworks.components import rnn_cell_impl as rnn_cell_impl_extended
from ops import capsule_initializer

from ntm_ops import create_linear_initializer
import collections

_alphabet_str = string.ascii_lowercase


class RecCapsuleCell(LayerRNNCell):
	""" Combination of RNN cell with capsule cell

	"""

	def __init__(self, num_capsules, capsule_dim, routing_iters,activation=None, input_probability_fn=None,
				 recurrent_probability_fn=None, kernel_initializer=None, logits_initializer=None, logits_prior=False,
				 reuse=None, name=None):
		super(RecCapsuleCell, self).__init__(_reuse=reuse, name=name)

		# For the moment expecting inputs to be 3-dimensional at every time step.
		# [batch_size x num_in_capsules X dim_in_capsules]
		self.input_spec = base_layer.InputSpec(ndim=3)

		self.num_capsules = num_capsules
		self.capsule_dim = capsule_dim
		self.kernel_initializer = kernel_initializer or capsule_initializer()
		self.logits_initializer = logits_initializer or tf.zeros_initializer()
		self.logits_prior = logits_prior
		self.routing_iters = routing_iters
		self._activation = activation or ops.squash
		self.input_probability_fn = input_probability_fn or tf.nn.softmax
		self.recurrent_probability_fn = recurrent_probability_fn or tf.nn.sigmoid

	@property
	def state_size(self):
		return tf.TensorShape([self.num_capsules, self.capsule_dim])

	@property
	def output_size(self):
		return tf.TensorShape([self.num_capsules, self.capsule_dim])

	def build(self, input_shapes):
		num_capsules_in = input_shapes[-2].value
		capsule_dim_in = input_shapes[-1].value

		if num_capsules_in is None:
			raise ValueError('number of input capsules must be defined')
		if capsule_dim_in is None:
			raise ValueError('input capsules dimension must be defined')

		self.input_kernel = self.add_variable(
			name='input_kernel',
			dtype=self.dtype,
			shape=[num_capsules_in, capsule_dim_in,
				   self.num_capsules, self.capsule_dim],
			initializer=self.kernel_initializer)

		self.state_kernel = self.add_variable(
			name='recurrent_kernel',
			dtype=self.dtype,
			shape=[self.num_capsules, self.capsule_dim,
				   self.num_capsules, self.capsule_dim],
			initializer=self.kernel_initializer)

		self.input_logits = self.add_variable(
			name='init_input_logits',
			dtype=self.dtype,
			shape=[num_capsules_in, self.num_capsules],
			initializer=self.logits_initializer,
			trainable=self.logits_prior)

		self.state_logits = self.add_variable(
			name='init_recurrent_logits',
			dtype=self.dtype,
			shape=[self.num_capsules, self.num_capsules],
			initializer=self.logits_initializer,
			trainable=self.logits_prior)

		self.built = True

	def call(self, inputs, state):
		"""
		apply the layer
		args:
		inputs: the inputs to the layer. the final two dimensions are
		  num_capsules_in and capsule_dim_in
		state: the recurrent inputs to the layer. the final two dimensions are
		  num_capsules and capsule_dim
		returns the output capsules as output capsule and as state. The last two dimensions are
		num_capsules and capsule_dim
		"""

		# compute the predictions from the inputs
		input_predictions, state_predictions, input_logits, state_logits = self.predict(inputs, state)

		# cluster the predictions
		outputs = self.cluster(input_predictions, state_predictions, input_logits, state_logits)

		return outputs, outputs

	def predict(self, inputs, state):
		"""
		compute the predictions for the output capsules and initialize the
		routing logits
		args:
		inputs: the inputs to the layer. the final two dimensions are
			num_capsules_in and capsule_dim_in
		 state: the recurrent inputs to the layer. the final two dimensions are
			num_capsules and capsule_dim
		 returns: the output capsule predictions
		 """

		with tf.name_scope('predict'):

			# number of shared dimensions. Assuming this is equal for inputs and state
			rank = len(inputs.shape)
			shared = rank-2

			if shared > 26-4:
				raise Exception('Not enough letters in the alphabet to use Einstein notation')

			# input_shape = [shared (typicaly batch size),Nin,Din], kernel_shape = [Nin, Din, Nout, Dout],
			# input_predictions_shape = [shared,Nin,Nout,Dout]
			shared_shape_str = _alphabet_str[0:shared]
			input_shape_str = shared_shape_str+'wx'
			kernel_shape_str = 'wxyz'
			output_shape_str = shared_shape_str+'wyz'
			ein_not = '%s,%s->%s' % (input_shape_str, kernel_shape_str, output_shape_str)

			input_predictions = tf.einsum(ein_not, inputs, self.input_kernel)
			state_predictions = tf.einsum(ein_not, state, self.state_kernel)

			# compute the logits for the inputs
			input_logits = self.input_logits
			for i in range(shared):
				if input_predictions.shape[shared-i-1].value is None:
					shape = tf.shape(input_predictions)[shared-i-1]
				else:
					shape = input_predictions.shape[shared-i-1].value
				tile = [shape] + [1]*len(input_logits.shape)
				input_logits = tf.tile(tf.expand_dims(input_logits, 0), tile)

			# compute the logits for the states
			state_logits = self.state_logits
			for i in range(shared):
				if state_predictions.shape[shared-i-1].value is None:
					shape = tf.shape(state_predictions)[shared-i-1]
				else:
					shape = state_predictions.shape[shared-i-1].value
				tile = [shape] + [1]*len(state_logits.shape)
				state_logits = tf.tile(tf.expand_dims(state_logits, 0), tile)

		return input_predictions, state_predictions, input_logits, state_logits

	def cluster(self, input_predictions, state_predictions, input_logits, state_logits):
		"""cluster the predictions into output capsules
		args:
		predictions: the predicted output capsules
		logits: the initial routing logits
		returns:
		the output capsules
		"""

		with tf.name_scope('cluster'):

			# define m-step
			def m_step(in_l, state_l):
				"""m step"""
				with tf.name_scope('m_step'):
					with tf.name_scope('m_step_in'):
						# compute the capsule contents
						in_w = self.input_probability_fn(in_l)
						in_caps = tf.reduce_sum(tf.expand_dims(in_w, -1)*input_predictions, -3)

					with tf.name_scope('m_step_state'):
						# compute the capsule contents
						state_w = self.recurrent_probability_fn(state_l)
						state_caps = tf.reduce_sum(tf.expand_dims(state_w, -1)*state_predictions, -3)

					capsules = in_caps + state_caps

				return capsules, in_caps, state_caps

			# define body of the while loop
			def body(in_l, state_l):
				"""body"""

				caps, _, _ = m_step(in_l, state_l)
				caps = self._activation(caps)

				# compare the capsule contents with the predictions
				in_similarity = tf.reduce_sum(input_predictions*tf.expand_dims(caps, -3), -1)
				state_similarity = tf.reduce_sum(state_predictions*tf.expand_dims(caps, -3), -1)

				return [in_l + in_similarity, state_l + state_similarity]

			# get the final logits with the while loop
			[in_lo, state_lo] = tf.while_loop(
				lambda l,ll: True,
				body, [input_logits, state_logits],
				maximum_iterations=self.routing_iters)

			# get the final output capsules
			capsules, _, _ = m_step(in_lo, state_lo)
			capsules = self._activation(capsules)

		return capsules


class RecCapsuleCellRecOnlyVote(RecCapsuleCell):
	""" Combination of RNN cell with capsule cell, the recurrent input is only used
	  for the voting process but not for the actual capsule activation

	"""

	def __init__(self, num_capsules, capsule_dim, routing_iters, activation=None,
				 input_probability_fn=None, recurrent_probability_fn=None,
				 kernel_initializer=None, logits_initializer=None, logits_prior=False,
				 accumulate_input_logits=True, accumulate_state_logits=True, reuse=None,
				 name=None):
		super(RecCapsuleCellRecOnlyVote, self).__init__(num_capsules, capsule_dim,
														routing_iters,activation,
														input_probability_fn, recurrent_probability_fn,
														kernel_initializer, logits_initializer,
														logits_prior=logits_prior, reuse=reuse, name=name)

		self.accumulate_input_logits = accumulate_input_logits
		self.accumulate_state_logits = accumulate_state_logits

	def cluster(self, input_predictions, state_predictions, input_logits, state_logits):
		"""cluster the predictions into output capsules
		args:
		predictions: the predicted output capsules
		logits: the initial routing logits
		returns:
		the output capsules
		"""

		with tf.name_scope('cluster'):

			# define m-step
			def m_step(in_l, state_l):
				"""m step"""
				with tf.name_scope('m_step'):
					with tf.name_scope('m_step_in'):
						# compute the capsule contents
						in_w = self.input_probability_fn(in_l)
						in_caps = tf.reduce_sum(
						tf.expand_dims(in_w, -1)*input_predictions, -3)

					with tf.name_scope('m_step_state'):
						# compute the capsule contents
						state_w = self.recurrent_probability_fn(state_l)
						state_caps = tf.reduce_sum(
						tf.expand_dims(state_w, -1)*state_predictions, -3)

				capsules = in_caps + state_caps

				return capsules, in_caps, state_caps

		# define body of the while loop
		def body(in_l, state_l, init_input_logits, init_state_logits):
			"""body"""

			caps, _, _ = m_step(in_l, state_l)
			caps = self._activation(caps)

			# compare the capsule contents with the predictions
			in_similarity = tf.reduce_sum(input_predictions*tf.expand_dims(caps, -3), -1)
			state_similarity = tf.reduce_sum(state_predictions*tf.expand_dims(caps, -3), -1)

			if self.accumulate_input_logits:
				in_l += in_similarity
			else:
				in_l = in_similarity + init_input_logits

			if self.accumulate_state_logits:
				state_l += state_similarity
			else:
				state_l = state_similarity + init_state_logits

			return [in_l, state_l, init_input_logits, init_state_logits]

		# get the final logits with the while loop
		[in_lo, state_lo, _, _] = tf.while_loop(
		  lambda l,ll,lll,llll: True,
		  body, [input_logits, state_logits, input_logits, state_logits],
		  maximum_iterations=self.routing_iters)

		# get the final output capsules, only using the input predictions!
		_, capsules, _ = m_step(in_lo, state_lo)
		capsules = self._activation(capsules)

		return capsules


class LSTMCapsuleCellRecOnlyVote(RecCapsuleCellRecOnlyVote):
	""" Combination of LSTM cell with capsule cell, the recurrent input is only used
	  for the voting process but not for the actual capsule activation

	"""

	def __init__(self, num_capsules, capsule_dim, routing_iters, forget_bias=1.0,
				 activation=None, input_probability_fn=None, recurrent_probability_fn=None,
				 kernel_initializer=None, logits_initializer=None, logits_prior=False,
				 accumulate_input_logits=True, accumulate_state_logits=True, gates_fc=False, reuse=None,
				 name=None):
		super(LSTMCapsuleCellRecOnlyVote, self).__init__(num_capsules, capsule_dim, routing_iters, activation,
														 input_probability_fn, recurrent_probability_fn,
														 kernel_initializer, logits_initializer, logits_prior,
														 accumulate_input_logits, accumulate_state_logits, reuse=reuse,
														 name=name)

		self.input_spec = base_layer.InputSpec(ndim=3)
		self._forget_bias = forget_bias
		if gates_fc:
			self.gates_output_shape = [3*self.num_capsules, self.capsule_dim]
		else:
			self.gates_output_shape = [3*self.num_capsules, 1]

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(tf.TensorShape([self.num_capsules, self.capsule_dim]),
											tf.TensorShape([self.num_capsules, self.capsule_dim]))

	@property
	def output_size(self):
		return tf.TensorShape([self.num_capsules, self.capsule_dim])

	def _linear(self, args, out_size):
		proj_size = args.get_shape()[-1]
		dtype = args.dtype
		weights = vs.get_variable("kernel_gates", [proj_size, out_size], dtype=dtype)
		out = math_ops.matmul(args, weights)
		bias = vs.get_variable("bias_gates", [out_size], dtype=dtype)
		out = nn_ops.bias_add(out, bias)

		return out

	def call(self, inputs, state):
		"""
		apply the layer
		args:
		inputs: the inputs to the layer. the final two dimensions are
		  num_capsules_in and capsule_dim_in
		state: the recurrent inputs to the layer. the final two dimensions are
		  num_capsules and capsule_dim
		returns the output capsules as output capsule and as state. The last two dimensions are
		num_capsules and capsule_dim
		"""

		batch_size = inputs.get_shape()[0]
		inputs_num_capsules = inputs.get_shape()[-2]
		inputs_caps_dim = inputs.get_shape()[-1]

		c, h = state

		# compute the predictions from the inputs
		input_predictions, state_predictions, input_logits, state_logits = self.predict(inputs, h)

		# cluster the predictions
		j = self.cluster(input_predictions, state_predictions, input_logits, state_logits)

		inputs_flat = tf.reshape(inputs, [batch_size, inputs_num_capsules*inputs_caps_dim])
		h_flat = tf.reshape(h, [batch_size, self.num_capsules*self.capsule_dim])

		args = array_ops.concat([inputs_flat, h_flat], 1)

		concat = self._linear(args, out_size=self.gates_output_shape[0]*self.gates_output_shape[1])
		concat = tf.reshape(concat, [batch_size]+self.gates_output_shape)

		i, f, o = array_ops.split(value=concat, num_or_size_splits=3, axis=1)

		g = self._activation(j)

		new_c = (c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)

		new_h = self._activation(new_c) * math_ops.sigmoid(o)

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)

		return new_h, new_state


class LSTMCapsuleCellRecOnlyVoteOutputMatrix(LSTMCapsuleCellRecOnlyVote):
	""" Combination of LSTM cell with capsule cell, the recurrent input is only used
	  for the voting process but not for the actual capsule activation. Instead of an output gate,
	  the activated capsule cell states are matrix multiplied and squashed.

	"""

	def __init__(self, num_capsules, capsule_dim, routing_iters, forget_bias=1.0,
				 activation=None, input_probability_fn=None, recurrent_probability_fn=None,
				 kernel_initializer=None, logits_initializer=None, logits_prior=False,
				 accumulate_input_logits=True, accumulate_state_logits=True, gates_fc=False, reuse=None,
				 name=None):
		super(LSTMCapsuleCellRecOnlyVoteOutputMatrix, self).__init__(num_capsules, capsule_dim, routing_iters,
																	   forget_bias, activation,
																	   input_probability_fn, recurrent_probability_fn,
																	   kernel_initializer, logits_initializer,
																	   logits_prior, accumulate_input_logits,
																	   accumulate_state_logits, gates_fc,
																	   reuse=reuse, name=name)

		if gates_fc:
			self.gates_output_shape = [2*self.num_capsules, self.capsule_dim]
		else:
			self.gates_output_shape = [2*self.num_capsules, 1]
		self.rot_matrix_shape = [self.num_capsules, self.capsule_dim, self.capsule_dim]

	def _linear(self, args, out_size):
		proj_size = args.get_shape()[-1]
		dtype = args.dtype
		weights = vs.get_variable("kernel_gates", [proj_size, out_size], dtype=dtype)
		out = math_ops.matmul(args, weights)
		bias = vs.get_variable("bias_gates", [out_size], dtype=dtype)
		out = nn_ops.bias_add(out, bias)

		return out

	def _linear_matrix(self, args, out_shape):
		proj_size = args.get_shape()[-1]
		dtype = args.dtype
		weights = vs.get_variable("kernel_output_matrix", [proj_size]+out_shape, dtype=dtype)
		out = tf.tensordot(args, weights, axes=[[-1], [0]])
		bias = vs.get_variable("bias_output_matrix", out_shape, dtype=dtype)
		out = out + tf.expand_dims(bias, 0)

		return out

	def call(self, inputs, state):
		"""
		apply the layer
		args:
		inputs: the inputs to the layer. the final two dimensions are
		  num_capsules_in and capsule_dim_in
		state: the recurrent inputs to the layer. the final two dimensions are
		  num_capsules and capsule_dim
		returns the output capsules as output capsule and as state. The last two dimensions are
		num_capsules and capsule_dim
		"""

		batch_size = inputs.get_shape()[0]
		inputs_num_capsules = inputs.get_shape()[-2]
		inputs_caps_dim = inputs.get_shape()[-1]

		c, h = state

		# compute the predictions from the inputs
		input_predictions, state_predictions, input_logits, state_logits = self.predict(inputs, h)

		# cluster the predictions
		j = self.cluster(input_predictions, state_predictions, input_logits, state_logits)

		inputs_flat = tf.reshape(inputs, [batch_size, inputs_num_capsules*inputs_caps_dim])
		h_flat = tf.reshape(h, [batch_size, self.num_capsules*self.capsule_dim])

		args = array_ops.concat([inputs_flat, h_flat], 1)

		concat = self._linear(args, out_size=self.gates_output_shape[0]*self.gates_output_shape[1])
		concat = tf.reshape(concat, [batch_size]+self.gates_output_shape)

		i, f = array_ops.split(value=concat, num_or_size_splits=2, axis=1)

		g = self._activation(j)

		new_c = (c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)

		o = self._linear_matrix(args, out_shape=self.rot_matrix_shape)

		activated_new_c = self._activation(new_c)

		# apply the matrix multiplication to the activated capsule cell states
		tmp = tf.squeeze(tf.matmul(o, tf.expand_dims(activated_new_c, -1)), -1)
		new_h = self._activation(tmp)

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)

		return new_h, new_state


class RNNCellLinearOut(BasicRNNCell):
	"""Same cell as BasicRNNCell, except that the activation function will
	only be applied to the recurrent output and not the feedforward output
	"""

	def call(self, inputs, state):
		gate_inputs = math_ops.matmul(
			array_ops.concat([inputs, state], 1), self._kernel)
		gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
		output = self._activation(gate_inputs)

		return gate_inputs, output


class LayerNormBasicLSTMCellLineairOut(LayerNormBasicLSTMCell):
	"""Same cell as BasicRNNCell, except that the activation function will
	only be applied to the recurrent output and not the feedforward output
	"""

	def call(self, inputs, state):
		"""LSTM cell with layer normalization and recurrent dropout."""
		c, h = state
		args = array_ops.concat([inputs, h], 1)
		concat = self._linear(args)
		dtype = args.dtype

		i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
		if self._layer_norm:
			i = self._norm(i, "input", dtype=dtype)
			j = self._norm(j, "transform", dtype=dtype)
			f = self._norm(f, "forget", dtype=dtype)
			o = self._norm(o, "output", dtype=dtype)

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

		new_c = (
			c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)
		if self._layer_norm:
			new_c = self._norm(new_c, "state", dtype=dtype)
		new_h = new_c * math_ops.sigmoid(o)

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return new_h, new_state


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class LeakGRUCell(LayerRNNCell):
	"""Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
	Args:
	num_units: int, The number of units in the GRU cell.
	activation: Nonlinearity to use.  Default: `tanh`.
	reuse: (optional) Python boolean describing whether to reuse variables
	 in an existing scope.  If not `True`, and the existing scope already has
	 the given variables, an error is raised.
	kernel_initializer: (optional) The initializer to use for the weight and
	projection matrices.
	bias_initializer: (optional) The initializer to use for the bias.
	name: String, the name of the layer. Layers with the same name will
	  share weights, but to avoid mistakes we require reuse=True in such
	  cases.
	"""

	def __init__(self,
				 num_units,
				 leak_factor=1.0,
				 activation=None,
				 reuse=None,
				 kernel_initializer=None,
				 bias_initializer=None,
				 name=None):
		super(LeakGRUCell, self).__init__(_reuse=reuse, name=name)

		# Inputs must be 2-dimensional.
		self.input_spec = base_layer.InputSpec(ndim=2)

		self._num_units = num_units
		self._leak_factor = leak_factor
		self._activation = activation or math_ops.tanh
		self._kernel_initializer = kernel_initializer
		self._bias_initializer = bias_initializer

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def build(self, inputs_shape):
		if inputs_shape[1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
						   % inputs_shape)

		input_depth = inputs_shape[1].value
		self._gate_kernel = self.add_variable(
			"gates/%s" % _WEIGHTS_VARIABLE_NAME,
			shape=[input_depth + self._num_units, 2 * self._num_units],
			initializer=self._kernel_initializer)
		self._gate_bias = self.add_variable(
			"gates/%s" % _BIAS_VARIABLE_NAME,
			shape=[2 * self._num_units],
			initializer=(
				self._bias_initializer
				if self._bias_initializer is not None
				else init_ops.constant_initializer(1.0, dtype=self.dtype)))
		self._candidate_kernel = self.add_variable(
			"candidate/%s" % _WEIGHTS_VARIABLE_NAME,
			shape=[input_depth + self._num_units, self._num_units],
			initializer=self._kernel_initializer)
		self._candidate_bias = self.add_variable(
			"candidate/%s" % _BIAS_VARIABLE_NAME,
			shape=[self._num_units],
			initializer=(
				self._bias_initializer
				if self._bias_initializer is not None
				else init_ops.zeros_initializer(dtype=self.dtype)))

		self.built = True

	def call(self, inputs, state):
		"""Gated recurrent unit (GRU) with nunits cells."""

		gate_inputs = math_ops.matmul(
			array_ops.concat([inputs, state], 1), self._gate_kernel)
		gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

		value = math_ops.sigmoid(gate_inputs)
		r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

		r_state = r * state

		candidate = math_ops.matmul(
			array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
		candidate = nn_ops.bias_add(candidate, self._candidate_bias)

		c = self._activation(candidate)
		new_h = u * state + (1 - u) * c
		new_h_recurrent = new_h * self._leak_factor
		return new_h, new_h_recurrent


class LayerNormBasicLeakLSTMCell(rnn_cell_impl.RNNCell):
	"""LSTM unit with layer normalization, recurrent dropout and
	memory leakage.

	This class adds layer normalization, recurrent dropout and memory
	leakage to a basic LSTM unit. Layer normalization implementation
	is based on:

	https://arxiv.org/abs/1607.06450.

	"Layer Normalization"
	Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

	and is applied before the internal nonlinearities.
	Recurrent dropout is base on:

	https://arxiv.org/abs/1603.05118

	"Recurrent Dropout without Memory Loss"
	Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
	"""

	def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0, input_size=None, activation=math_ops.tanh,
				 layer_norm=True, norm_gain=1.0, norm_shift=0.0, dropout_keep_prob=1.0, dropout_prob_seed=None,
				 reuse=None):
		"""Initializes the basic LSTM cell.

		Args:
		  num_units: int, The number of units in the LSTM cell.
		  leak_factor:
		  forget_bias: float, The bias added to forget gates (see above).
		  input_size: Deprecated and unused.
		  activation: Activation function of the inner states.
		  layer_norm: If `True`, layer normalization will be applied.
		  norm_gain: float, The layer normalization gain initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  norm_shift: float, The layer normalization shift initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
			recurrent dropout probability value. If float and 1.0, no dropout will
			be applied.
		  dropout_prob_seed: (optional) integer, the randomness seed.
		  reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.
		"""
		super(LayerNormBasicLeakLSTMCell, self).__init__(_reuse=reuse)

		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated.", self)

		self._num_units = num_units
		self._leak_factor = leak_factor
		self._activation = activation
		self._forget_bias = forget_bias
		self._keep_prob = dropout_keep_prob
		self._seed = dropout_prob_seed
		self._layer_norm = layer_norm
		self._g = norm_gain
		self._b = norm_shift
		self._reuse = reuse

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def _norm(self, inp, scope):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(self._g)
		beta_init = init_ops.constant_initializer(self._b)
		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init)
			vs.get_variable("beta", shape=shape, initializer=beta_init)
		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def _linear(self, args):
		out_size = 4 * self._num_units
		proj_size = args.get_shape()[-1]
		weights = vs.get_variable("kernel", [proj_size, out_size])
		out = math_ops.matmul(args, weights)
		if not self._layer_norm:
			bias = vs.get_variable("bias", [out_size])
			out = nn_ops.bias_add(out, bias)
		return out

	def call(self, inputs, state):
		"""LSTM cell with layer normalization and recurrent dropout."""
		c, h = state
		args = array_ops.concat([inputs, h], 1)
		concat = self._linear(args)

		i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
		if self._layer_norm:
			i = self._norm(i, "input")
			j = self._norm(j, "transform")
			f = self._norm(f, "forget")
			o = self._norm(o, "output")

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

		new_c = (c * math_ops.sigmoid(f + self._forget_bias)
				 + math_ops.sigmoid(i) * g)
		if self._layer_norm:
			new_c = self._norm(new_c, "state")
		new_h = self._activation(new_c) * math_ops.sigmoid(o)

		new_c = new_c * self._leak_factor

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return new_h, new_state


class LayerNormIZNotRecLeakLSTMCell(rnn_cell_impl.RNNCell):
	"""LSTM unit with layer normalization, recurrent dropout and
	memory leakage. The input gate (i) and the transformed input (z, or j as
	in the code below) are only based on the layers input and not recurrent

	This class adds layer normalization, recurrent dropout and memory
	leakage to a basic LSTM unit. Layer normalization implementation
	is based on:

	https://arxiv.org/abs/1607.06450.

	"Layer Normalization"
	Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

	and is applied before the internal nonlinearities.
	Recurrent dropout is base on:

	https://arxiv.org/abs/1603.05118

	"Recurrent Dropout without Memory Loss"
	Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
	"""

	def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0, input_size=None, activation=math_ops.tanh,
				 layer_norm=True, norm_gain=1.0, norm_shift=0.0, dropout_keep_prob=1.0, dropout_prob_seed=None,
				 reuse=None):
		"""Initializes the basic LSTM cell.

		Args:
		  num_units: int, The number of units in the LSTM cell.
		  leak_factor:
		  forget_bias: float, The bias added to forget gates (see above).
		  input_size: Deprecated and unused.
		  activation: Activation function of the inner states.
		  layer_norm: If `True`, layer normalization will be applied.
		  norm_gain: float, The layer normalization gain initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  norm_shift: float, The layer normalization shift initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
			recurrent dropout probability value. If float and 1.0, no dropout will
			be applied.
		  dropout_prob_seed: (optional) integer, the randomness seed.
		  reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.
		"""
		super(LayerNormIZNotRecLeakLSTMCell, self).__init__(_reuse=reuse)

		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated.", self)

		self._num_units = num_units
		self._leak_factor = leak_factor
		self._activation = activation
		self._forget_bias = forget_bias
		self._keep_prob = dropout_keep_prob
		self._seed = dropout_prob_seed
		self._layer_norm = layer_norm
		self._g = norm_gain
		self._b = norm_shift
		self._reuse = reuse

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def _norm(self, inp, scope):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(self._g)
		beta_init = init_ops.constant_initializer(self._b)
		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init)
			vs.get_variable("beta", shape=shape, initializer=beta_init)
		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def _linear(self, args):
		out_size = 2 * self._num_units
		proj_size = args.get_shape()[-1]
		weights = vs.get_variable("kernel", [proj_size, out_size])
		out = math_ops.matmul(args, weights)
		if not self._layer_norm:
			bias = vs.get_variable("bias", [out_size])
			out = nn_ops.bias_add(out, bias)
		return out

	def _linear_inonly(self, args):
		out_size = 2 * self._num_units
		proj_size = args.get_shape()[-1]
		weights = vs.get_variable("kernel_inonly", [proj_size, out_size])
		out = math_ops.matmul(args, weights)
		if not self._layer_norm:
			bias = vs.get_variable("bias_inonly", [out_size])
			out = nn_ops.bias_add(out, bias)
		return out

	def call(self, inputs, state):
		"""LSTM cell with layer normalization and recurrent dropout."""
		c, h = state
		args = array_ops.concat([inputs, h], 1)
		concat = self._linear(args)
		concat_inonly = self._linear_inonly(inputs)

		f, o = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
		i, j = array_ops.split(value=concat_inonly, num_or_size_splits=2, axis=1)
		if self._layer_norm:
			i = self._norm(i, "input")
			j = self._norm(j, "transform")
			f = self._norm(f, "forget")
			o = self._norm(o, "output")

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

		new_c = (c * math_ops.sigmoid(f + self._forget_bias)
				 + math_ops.sigmoid(i) * g)
		if self._layer_norm:
			new_c = self._norm(new_c, "state")
		new_h = self._activation(new_c) * math_ops.sigmoid(o)

		new_c = new_c * self._leak_factor

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return new_h, new_state


class LayerNormNotRecLeakLSTMCell(rnn_cell_impl.RNNCell):
	"""LSTM unit with layer normalization, recurrent dropout and
	memory leakage. All weights are only dependent on cell inputs and not
	on recurrent hidden units

	This class adds layer normalization, recurrent dropout and memory
	leakage to a basic LSTM unit. Layer normalization implementation
	is based on:

	https://arxiv.org/abs/1607.06450.

	"Layer Normalization"
	Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

	and is applied before the internal nonlinearities.
	Recurrent dropout is base on:

	https://arxiv.org/abs/1603.05118

	"Recurrent Dropout without Memory Loss"
	Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
	"""

	def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0, input_size=None, activation=math_ops.tanh,
				 layer_norm=True, norm_gain=1.0, norm_shift=0.0, dropout_keep_prob=1.0, dropout_prob_seed=None,
				 reuse=None):
		"""Initializes the basic LSTM cell.

		Args:
		  num_units: int, The number of units in the LSTM cell.
		  leak_factor:
		  forget_bias: float, The bias added to forget gates (see above).
		  input_size: Deprecated and unused.
		  activation: Activation function of the inner states.
		  layer_norm: If `True`, layer normalization will be applied.
		  norm_gain: float, The layer normalization gain initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  norm_shift: float, The layer normalization shift initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
			recurrent dropout probability value. If float and 1.0, no dropout will
			be applied.
		  dropout_prob_seed: (optional) integer, the randomness seed.
		  reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.
		"""
		super(LayerNormNotRecLeakLSTMCell, self).__init__(_reuse=reuse)

		if input_size is not None:
		  logging.warn("%s: The input_size parameter is deprecated.", self)

		self._num_units = num_units
		self._leak_factor = leak_factor
		self._activation = activation
		self._forget_bias = forget_bias
		self._keep_prob = dropout_keep_prob
		self._seed = dropout_prob_seed
		self._layer_norm = layer_norm
		self._g = norm_gain
		self._b = norm_shift
		self._reuse = reuse

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def _norm(self, inp, scope):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(self._g)
		beta_init = init_ops.constant_initializer(self._b)
		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init)
			vs.get_variable("beta", shape=shape, initializer=beta_init)
		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def _linear(self, args):
		out_size = 4 * self._num_units
		proj_size = args.get_shape()[-1]
		weights = vs.get_variable("kernel", [proj_size, out_size])
		out = math_ops.matmul(args, weights)
		if not self._layer_norm:
			bias = vs.get_variable("bias", [out_size])
			out = nn_ops.bias_add(out, bias)
		return out

	def call(self, inputs, state):
		"""LSTM cell with layer normalization and recurrent dropout."""
		c, h = state
		# args = array_ops.concat([inputs, h], 1)
		concat = self._linear(inputs)

		i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
		if self._layer_norm:
			i = self._norm(i, "input")
			j = self._norm(j, "transform")
			f = self._norm(f, "forget")
			o = self._norm(o, "output")

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

		new_c = (c * math_ops.sigmoid(f + self._forget_bias)
				 + math_ops.sigmoid(i) * g)
		if self._layer_norm:
			new_c = self._norm(new_c, "state")
		new_h = self._activation(new_c) * math_ops.sigmoid(o)

		new_c = new_c * self._leak_factor

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return new_h, new_state


class LayerNormBasicLeakchLSTMCell(rnn_cell_impl.RNNCell):
	"""LSTM unit with layer normalization, recurrent dropout and
	memory leakage, both the c and h variable.

	This class adds layer normalization, recurrent dropout and memory
	leakage to a basic LSTM unit. Layer normalization implementation
	is based on:

	https://arxiv.org/abs/1607.06450.

	"Layer Normalization"
	Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

	and is applied before the internal nonlinearities.
	Recurrent dropout is base on:

	https://arxiv.org/abs/1603.05118

	"Recurrent Dropout without Memory Loss"
	Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
	"""

	def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0, input_size=None, activation=math_ops.tanh,
				 layer_norm=True, norm_gain=1.0, norm_shift=0.0, dropout_keep_prob=1.0, dropout_prob_seed=None,
				 reuse=None):
		"""Initializes the basic LSTM cell.

		Args:
		  num_units: int, The number of units in the LSTM cell.
		  leak_factor:
		  forget_bias: float, The bias added to forget gates (see above).
		  input_size: Deprecated and unused.
		  activation: Activation function of the inner states.
		  layer_norm: If `True`, layer normalization will be applied.
		  norm_gain: float, The layer normalization gain initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  norm_shift: float, The layer normalization shift initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
			recurrent dropout probability value. If float and 1.0, no dropout will
			be applied.
		  dropout_prob_seed: (optional) integer, the randomness seed.
		  reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.
		"""
		super(LayerNormBasicLeakchLSTMCell, self).__init__(_reuse=reuse)

		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated.", self)

		self._num_units = num_units
		self._leak_factor = leak_factor
		self._activation = activation
		self._forget_bias = forget_bias
		self._keep_prob = dropout_keep_prob
		self._seed = dropout_prob_seed
		self._layer_norm = layer_norm
		self._g = norm_gain
		self._b = norm_shift
		self._reuse = reuse

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def _norm(self, inp, scope):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(self._g)
		beta_init = init_ops.constant_initializer(self._b)
		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init)
			vs.get_variable("beta", shape=shape, initializer=beta_init)
		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def _linear(self, args):
		out_size = 4 * self._num_units
		proj_size = args.get_shape()[-1]
		weights = vs.get_variable("kernel", [proj_size, out_size])
		out = math_ops.matmul(args, weights)
		if not self._layer_norm:
			bias = vs.get_variable("bias", [out_size])
			out = nn_ops.bias_add(out, bias)
		return out

	def call(self, inputs, state):
		"""LSTM cell with layer normalization and recurrent dropout."""
		c, h = state

		# leak recurrent memory
		c = c * self._leak_factor
		h = h * self._leak_factor

		args = array_ops.concat([inputs, h], 1)
		concat = self._linear(args)

		i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
		if self._layer_norm:
			i = self._norm(i, "input")
			j = self._norm(j, "transform")
			f = self._norm(f, "forget")
			o = self._norm(o, "output")

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

		new_c = (c * math_ops.sigmoid(f + self._forget_bias)
				 + math_ops.sigmoid(i) * g)
		if self._layer_norm:
			new_c = self._norm(new_c, "state")
		new_h = self._activation(new_c) * math_ops.sigmoid(o)

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return new_h, new_state


class LayerNormResetLSTMCell(rnn_cell_impl_extended.RNNCell):
	"""LSTM unit were state is reset every k steps
	"""

	def __init__(self,
				 num_units,
				 t_reset=1,
				 forget_bias=1.0,
				 input_size=None,
				 activation=math_ops.tanh,
				 layer_norm=True,
				 norm_gain=1.0,
				 norm_shift=0.0,
				 dropout_keep_prob=1.0,
				 dropout_prob_seed=None,
				 reuse=None):
		"""Initializes the reset LSTM cell.
		Args:
		  num_units: int, The number of units in the LSTM cell.
		  t_reset: int, the reset cycle period.
		  forget_bias: float, The bias added to forget gates (see above).
		  input_size: Deprecated and unused.
		  activation: Activation function of the inner states.
		  layer_norm: If `True`, layer normalization will be applied.
		  norm_gain: float, The layer normalization gain initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  norm_shift: float, The layer normalization shift initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
			recurrent dropout probability value. If float and 1.0, no dropout will
			be applied.
		  dropout_prob_seed: (optional) integer, the randomness seed.
		  reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.
		"""
		super(LayerNormResetLSTMCell, self).__init__(_reuse=reuse)

		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated.", self)

		self._num_units = num_units
		self._t_reset = t_reset
		self._activation = activation
		self._forget_bias = forget_bias
		self._keep_prob = dropout_keep_prob
		self._seed = dropout_prob_seed
		self._layer_norm = layer_norm
		self._norm_gain = norm_gain
		self._norm_shift = norm_shift
		self._reuse = reuse

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(tf.TensorShape([self._t_reset, self._num_units]), tf.TensorShape([self._t_reset, self._num_units]))

	@property
	# output tuple: (actual output, multi state output)
	def output_size(self):
		return self._num_units, tf.TensorShape([self._t_reset, self._num_units])

	def _norm(self, inp, scope, dtype=dtypes.float32):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(self._norm_gain)
		beta_init = init_ops.constant_initializer(self._norm_shift)
		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
			vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def _linear(self, args):
		out_size = 4 * self._num_units
		proj_size = args.get_shape()[-1]
		dtype = args.dtype
		weights = vs.get_variable("kernel", [proj_size, out_size], dtype=dtype)
		out = tf.tensordot(args, weights, [[-1], [0]])
		if not self._layer_norm:
			bias = vs.get_variable("bias", [out_size], dtype=dtype)
			out = nn_ops.bias_add(out, bias)
		return out

	def call(self, inputs, state, time):
		"""LSTM cell with layer normalization and recurrent dropout."""

		state_index = tf.mod(time, self._t_reset)
		next_state_index = tf.mod(time + 1, self._t_reset)
		c, h = state

		# apply reset
		tmp = 1-tf.scatter_nd(tf.expand_dims(tf.expand_dims(state_index, 0), 0), tf.constant([1.0]),
							  tf.constant([self._t_reset]))
		reset_mask = tf.expand_dims(tf.expand_dims(tmp, 0), -1)

		c_reset = c * reset_mask
		h_reset = h * reset_mask

		args = array_ops.concat([inputs, h_reset], -1)
		concat = self._linear(args)
		dtype = args.dtype

		i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=-1)
		if self._layer_norm:
			i = self._norm(i, "input", dtype=dtype)
			j = self._norm(j, "transform", dtype=dtype)
			f = self._norm(f, "forget", dtype=dtype)
			o = self._norm(o, "output", dtype=dtype)

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)
		# (i,g,f,o) = (tf.expand_dims(val, -1) for val in (i,g,f,o))

		new_c = (c_reset * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)
		if self._layer_norm:
			new_c = self._norm(new_c, "state", dtype=dtype)
		new_h = self._activation(new_c) * math_ops.sigmoid(o)
		new_h_maxmem = tf.gather(new_h, next_state_index, axis=1)

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return (new_h_maxmem, new_h), new_state


class LayerNormGroupResetLSTMCell(LayerNormResetLSTMCell):
	"""LSTM unit were state is reset every k steps, for 1 group of units
	"""

	def __init__(self,
				 num_units,
				 t_reset=1,
				 group_size=1,
				 forget_bias=1.0,
				 input_size=None,
				 activation=math_ops.tanh,
				 layer_norm=True,
				 norm_gain=1.0,
				 norm_shift=0.0,
				 dropout_keep_prob=1.0,
				 dropout_prob_seed=None,
				 reuse=None):
		"""Initializes the reset LSTM cell.
		Args:
		  num_units: int, The number of units in the LSTM cell.
		  t_reset: int, the reset cycle period.
		  group_size: int, the size of the group
		  forget_bias: float, The bias added to forget gates (see above).
		  input_size: Deprecated and unused.
		  activation: Activation function of the inner states.
		  layer_norm: If `True`, layer normalization will be applied.
		  norm_gain: float, The layer normalization gain initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  norm_shift: float, The layer normalization shift initial value. If
			`layer_norm` has been set to `False`, this argument will be ignored.
		  dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
			recurrent dropout probability value. If float and 1.0, no dropout will
			be applied.
		  dropout_prob_seed: (optional) integer, the randomness seed.
		  reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.
		"""
		super(LayerNormGroupResetLSTMCell, self).__init__(num_units, t_reset=t_reset, forget_bias=forget_bias,
														  input_size=input_size, activation=activation,
														  layer_norm=layer_norm, norm_gain=norm_gain,
														  norm_shift=norm_shift, dropout_keep_prob=dropout_keep_prob,
														  dropout_prob_seed=dropout_prob_seed, reuse=reuse)

		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated.", self)

		self._group_size = group_size
		num_replicates = float(self._t_reset) / float(self._group_size)
		self._num_replicates = int(num_replicates)

		if self._num_replicates != num_replicates:
			raise ValueError('t_reset should be a multiple of group_size')

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(tf.TensorShape([self._num_replicates, self._num_units]),
											tf.TensorShape([self._num_replicates, self._num_units]))

	@property
	# output tuple: (actual output, multi state output)
	def output_size(self):
		return self._num_units, tf.TensorShape([self._num_replicates, self._num_units])

	def call(self, inputs, state, time):
		"""LSTM cell with layer normalization and recurrent dropout."""

		state_index_in_group = tf.mod(time, self._group_size)
		group_index = tf.floor_div(time, self._group_size)
		replicate_index = tf.mod(group_index, self._num_replicates)
		next_group_index = tf.floor_div(time + 1, self._group_size)
		next_replicate_index = tf.mod(next_group_index, self._num_replicates)

		c, h = state

		# here we reset the correct state (but only if we reached the end of the group)
		tmp = 1-tf.scatter_nd(tf.expand_dims(tf.expand_dims(replicate_index, 0), 0), tf.constant([1.0]),
							  tf.constant([self._num_replicates]))
		reset_mask = tf.expand_dims(tf.expand_dims(tmp, 0), -1)

		reset_flag = tf.equal(state_index_in_group+1, self._group_size)
		c_reset = tf.cond(reset_flag, lambda: c * reset_mask, lambda: c)
		h_reset = tf.cond(reset_flag, lambda: h * reset_mask, lambda: h)

		args = array_ops.concat([inputs, h_reset], -1)
		concat = self._linear(args)
		dtype = args.dtype

		i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=-1)
		if self._layer_norm:
			i = self._norm(i, "input", dtype=dtype)
			j = self._norm(j, "transform", dtype=dtype)
			f = self._norm(f, "forget", dtype=dtype)
			o = self._norm(o, "output", dtype=dtype)

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)
		# (i,g,f,o) = (tf.expand_dims(val, -1) for val in (i,g,f,o))

		new_c = (
			c_reset * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)
		if self._layer_norm:
			new_c = self._norm(new_c, "state", dtype=dtype)
		new_h = self._activation(new_c) * math_ops.sigmoid(o)
		new_h_maxmem = tf.gather(new_h, next_replicate_index, axis=1)

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return (new_h_maxmem, new_h), new_state


NTMControllerState = collections.namedtuple('NTMState', ('read_vector', 'w', 'M'))


class NTMCell(tf.contrib.rnn.RNNCell):
	"""Based on https://github.com/MarkPKCollier/NeuralTuringMachine"""
	def __init__(
			self, input_size, memory_size, memory_vector_dim, read_head_num, write_head_num, addressing_mode='content_and_location',
			shift_win_size=3, clip_value=20, init_mode='constant', init_mode_other_params='constant_other_params', reuse=False):
		self.input_size = input_size
		self.memory_size = memory_size
		self.memory_vector_dim = memory_vector_dim
		self.read_head_num = read_head_num
		self.write_head_num = write_head_num
		self.addressing_mode = addressing_mode
		self.reuse = reuse
		self.clip_value = clip_value

		self.init_mode = init_mode
		self.init_mode_other_params = init_mode_other_params

		self.output_dim = [read_head_num, memory_vector_dim]
		self.read_head_dim = [read_head_num, memory_size]
		self.shift_win_size = np.min([shift_win_size, self.memory_size])

		self.o2p_initializer = create_linear_initializer(self.input_size)

	def __call__(self, inputs, prev_state):
		batch_size = int(inputs.get_shape()[0])
		prev_read_vector = prev_state.read_vector

		combined_inputs = tf.concat(
			[inputs, tf.reshape(prev_read_vector, [batch_size, self.read_head_num * self.memory_vector_dim])], axis=1)

		num_parameters_per_head = self.memory_vector_dim + 1 + 1 + self.shift_win_size + 1
		num_heads = self.read_head_num + self.write_head_num
		num_parameters_erase_per_write_head = self.memory_vector_dim
		total_parameter_num = num_parameters_per_head * num_heads + num_parameters_erase_per_write_head * 2 * self.write_head_num
		with tf.variable_scope("o2p", self.reuse):
			parameters = tf.contrib.layers.fully_connected(
				combined_inputs, total_parameter_num, activation_fn=None,
				weights_initializer=self.o2p_initializer)
			parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)
		head_parameters = parameters[:, :num_parameters_per_head * num_heads]
		head_parameters = tf.reshape(head_parameters, [batch_size, num_heads, num_parameters_per_head])
		erase_parameters = parameters[:, num_parameters_per_head * num_heads:
										 num_parameters_per_head * num_heads + num_parameters_erase_per_write_head * self.write_head_num]
		erase_parameters = tf.reshape(erase_parameters, [batch_size, self.write_head_num, num_parameters_erase_per_write_head])
		add_parameters = parameters[:, num_parameters_per_head * num_heads + num_parameters_erase_per_write_head * self.write_head_num:]
		add_parameters = tf.reshape(add_parameters, [batch_size, self.write_head_num, num_parameters_erase_per_write_head])

		prev_w = prev_state.w
		prev_M = prev_state.M

		k = tf.tanh(head_parameters[:, :, 0:self.memory_vector_dim])
		beta = tf.nn.softplus(head_parameters[:, :, self.memory_vector_dim])
		g = tf.sigmoid(head_parameters[:, :, self.memory_vector_dim + 1])
		s = tf.nn.softmax(
			head_parameters[:, :, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + self.shift_win_size], -1)
		gamma = tf.nn.softplus(head_parameters[:, :, -1]) + 1

		w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w)

		read_w = tf.expand_dims(w[:, :self.read_head_num], -1)
		read_vector = tf.reduce_sum(read_w * tf.expand_dims(prev_M, 1), axis=2)

		# Writing (Sec 3.2)

		M = prev_M
		write_w = tf.expand_dims(w[:, self.read_head_num:], -1)
		forget_vectors = tf.reduce_prod(1-tf.matmul(write_w, tf.expand_dims(tf.sigmoid(erase_parameters), 2)), 1)
		add_vectors = tf.reduce_sum(tf.matmul(write_w, tf.expand_dims(tf.tanh(add_parameters), 2)), 1)
		M = M * forget_vectors + add_vectors

		read_w = tf.squeeze(read_w, -1)

		return (read_vector, read_w), NTMControllerState(read_vector=read_vector, w=w, M=M)

	def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):
		num_head = prev_w.get_shape()[1]
		# Sec 3.3.1 Focusing by Content

		# Cosine Similarity

		k = tf.expand_dims(k, axis=-1)
		k_norm = tf.norm(k, axis=2, keepdims=True)
		k_normalized = k/(k_norm + 1e-12)
		prev_M = tf.tile(tf.expand_dims(prev_M, 1), [1, num_head, 1, 1])
		prev_M_norm = tf.norm(prev_M, axis=-1, keepdims=True)
		prev_M_normalized = prev_M/(prev_M_norm + 1e-12)

		K = tf.squeeze(tf.matmul(prev_M_normalized, k_normalized), -1)			                # eq (6)

		# Calculating w^c

		K_amplified = tf.exp(tf.expand_dims(beta, axis=2) * K)
		w_c = tf.nn.softmax(K_amplified, axis=2)								# eq (5)

		if self.addressing_mode == 'content':                                   # Only focus on content
			return w_c

		# Sec 3.3.2 Focusing by Location

		g = tf.expand_dims(g, axis=2)
		w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

		if self.addressing_mode == 'content_gated':                     # Only focus on content and gated with previous
			return w_g

		# s shift coefficients for (i-j; eq.8) are in the order 0,-1,...,-shift_range,+shift_range,...,+1
		shift_range = self.shift_win_size/2  # round down for odd size
		s = tf.concat(
			[
				s[:, :, :shift_range + 1],
				tf.zeros(s.get_shape()[0:2].concatenate(tf.TensorShape([self.memory_size - self.shift_win_size]))),
				s[:, :, shift_range+1:]],
			axis=-1)
		t = tf.concat([tf.reverse(s, axis=[-1]), tf.reverse(s, axis=[-1])], axis=-1)
		s_matrix = tf.stack(
			[t[:, :, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
			axis=-1
		)
		w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=2) * s_matrix, axis=-1)      # eq (8)
		w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=-1))
		w = w_sharpen / tf.reduce_sum(w_sharpen, axis=-1, keep_dims=True)        # eq (9)

		return w

	def zero_state(self, batch_size, dtype):
		with tf.variable_scope('init', reuse=self.reuse):
			read_vector_init = tf.tanh(
				tf.get_variable(
					name='init_read_vector', shape=self.memory_vector_dim,
					initializer=create_linear_initializer(self.memory_vector_dim),
					trainable=self.init_mode_other_params == 'learned')
			)
			read_vector_init = tf.tile(tf.expand_dims(tf.expand_dims(read_vector_init, 0), 0), [batch_size, self.read_head_num, 1])

			w_init = tf.nn.softmax(
				tf.get_variable(
					name='init_w', shape=[self.read_head_num + self.write_head_num, self.memory_size],
					initializer=create_linear_initializer(self.memory_size),
					trainable=self.init_mode_other_params == 'learned'),
				-1)
			w_init = tf.tile(tf.expand_dims(w_init, 0), [batch_size, 1, 1])

			if self.init_mode == 'learned':
				M = tf.tanh(tf.get_variable(
					name='init_M', shape=[self.memory_size, self.memory_vector_dim],
					initializer=create_linear_initializer(self.memory_vector_dim)))
			elif self.init_mode == 'random':
				M = tf.random_normal(shape=[self.memory_size, self.memory_vector_dim], mean=0.0, stddev=0.5, name='init_M')
			elif self.init_mode == 'constant':
				M = tf.get_variable(
					name='init_M', shape=[self.memory_size, self.memory_vector_dim],
					initializer=tf.constant_initializer(1e-6), trainable=False)
			else:
				raise Exception('Unknown initialization method')
			M = tf.tile(tf.expand_dims(M, 0), [batch_size, 1, 1])

			return NTMControllerState(
				read_vector=read_vector_init,
				w=w_init,
				M=M)

	@property
	def state_size(self):
		return NTMControllerState(
			read_vector=[self.read_head_num, self.memory_vector_dim],
			w=[self.read_head_num + self.write_head_num, self.memory_size],
			M=tf.TensorShape([self.memory_size * self.memory_vector_dim]))

	@property
	def output_size(self):
		return tf.TensorShape(self.output_dim), tf.TensorShape(self.read_head_dim)


class ScopeRNNCellWrapper(tf.contrib.rnn.RNNCell):
	"""this wraps an RNN cell to make sure it uses the same scope every time it's called"""

	def __init__(self, cell, name):
		"""ScopeRNNCellWrapper constructor"""

		self._cell = cell
		self.scope = tf.VariableScope(None, name)

	@property
	def output_size(self):
		"""return cell output size"""

		return self._cell.output_size

	@property
	def state_size(self):
		"""return cell state size"""

		return self._cell.state_size

	def zero_state(self, batch_size, dtype):
		"""the cell zero state"""

		return self._cell.zero_state(batch_size, dtype)

	def __call__(self, inputs, state, scope=None):
		"""call wrapped cell with constant scope"""

		with tf.variable_scope(self.scope):
			output, new_state = self._cell(inputs, state, scope)

		return output, new_state


class StateOutputWrapper(tf.contrib.rnn.RNNCell):
	"""this wraps an RNN cell to make it output its concatenated state instead of the output"""

	def __init__(self, cell):
		"""StateOutputWrapper constructor"""

		self._cell = cell

	@property
	def output_size(self):
		"""return cell output size"""

		return sum([int(x) for x in nest.flatten(self._cell.state_size)])

	@property
	def state_size(self):
		"""return cell state size"""

		return self._cell.state_size

	def zero_state(self, batch_size, dtype):
		"""the cell zero state"""

		return self._cell.zero_state(batch_size, dtype)

	def __call__(self, inputs, state, scope=None):
		"""call wrapped cell with constant scope"""

		_, new_state = self._cell(inputs, state, scope)
		output = tf.concat(nest.flatten(new_state), axis=1)

		return output, new_state
