# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module implementing RNN Cells.
This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables

from tensorflow.python.ops import rnn_cell_impl

import pdb


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class RNNCell(base_layer.Layer):
	"""Abstract object representing an RNN cell.
	Every `RNNCell` must have the properties below and implement `call` with
	the signature `(output, next_state) = call(input, state)`.  The optional
	third input argument, `scope`, is allowed for backwards compatibility
	purposes; but should be left off for new subclasses.
	This definition of cell differs from the definition used in the literature.
	In the literature, 'cell' refers to an object with a single scalar output.
	This definition refers to a horizontal array of such units.
	An RNN cell, in the most abstract setting, is anything that has
	a state and performs some operation that takes a matrix of inputs.
	This operation results in an output matrix with `self.output_size` columns.
	If `self.state_size` is an integer, this operation also results in a new
	state matrix with `self.state_size` columns.  If `self.state_size` is a
	(possibly nested tuple of) TensorShape object(s), then it should return a
	matching structure of Tensors having shape `[batch_size].concatenate(s)`
	for each `s` in `self.batch_size`.
	"""

	def __call__(self, inputs, state, time, scope=None):
		"""Run this RNN cell on inputs, starting from the given state.
		Args:
			inputs: `2-D` tensor with shape `[batch_size, input_size]`.
			state: if `self.state_size` is an integer, this should be a `2-D Tensor`
			with shape `[batch_size, self.state_size]`.  Otherwise, if
			`self.state_size` is a tuple of integers, this should be a tuple
			with shapes `[batch_size, s] for s in self.state_size`.
			time: `0-D` tensor with shape indicating the current time in the rnn loop.
			scope: VariableScope for the created subgraph; defaults to class name.
		Returns:
			A pair containing:
				- Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
				- New state: Either a single `2-D` tensor, or a tuple of tensors matching
				the arity and shapes of `state`.
		"""
		if scope is not None:
			with vs.variable_scope(scope, custom_getter=self._rnn_get_variable) as scope:
				return super(RNNCell, self).__call__(inputs, state, time, scope=scope)
		else:
			scope_attrname = "rnncell_scope"
			scope = getattr(self, scope_attrname, None)
			if scope is None:
				scope = vs.variable_scope(vs.get_variable_scope(), custom_getter=self._rnn_get_variable)
				setattr(self, scope_attrname, scope)
			with scope:
				return super(RNNCell, self).__call__(inputs, state, time)

	def _rnn_get_variable(self, getter, *args, **kwargs):
		variable = getter(*args, **kwargs)
		if context.executing_eagerly():
			trainable = variable._trainable  # pylint: disable=protected-access
		else:
			trainable = (
				variable in tf_variables.trainable_variables() or
				(
						isinstance(variable, tf_variables.PartitionedVariable) and
						list(variable)[0] in tf_variables.trainable_variables()))
		if trainable and variable not in self._trainable_weights:
			self._trainable_weights.append(variable)
		elif not trainable and variable not in self._non_trainable_weights:
			self._non_trainable_weights.append(variable)
		return variable

	@property
	def state_size(self):
		"""size(s) of state(s) used by this cell.
		It can be represented by an Integer, a TensorShape or a tuple of Integers
		or TensorShapes.
		"""
		raise NotImplementedError("Abstract method")

	@property
	def output_size(self):
		"""Integer or TensorShape: size of outputs produced by this cell."""
		raise NotImplementedError("Abstract method")

	def build(self, _):
		# This tells the parent Layer object that it's OK to call
		# self.add_variable() inside the call() method.
		pass

	def zero_state(self, batch_size, dtype):
		"""Return zero-filled state tensor(s).
		Args:
			batch_size: int, float, or unit Tensor representing the batch size.
			dtype: the data type to use for the state.
		Returns:
			If `state_size` is an int or TensorShape, then the return value is a
			`N-D` tensor of shape `[batch_size, state_size]` filled with zeros.
			If `state_size` is a nested list or tuple, then the return value is
			a nested list or tuple (of the same structure) of `2-D` tensors with
			the shapes `[batch_size, s]` for each s in `state_size`.
		"""
		# Try to use the last cached zero_state. This is done to avoid recreating
		# zeros, especially when eager execution is enabled.
		state_size = self.state_size
		is_eager = context.executing_eagerly()
		if is_eager and hasattr(self, "_last_zero_state"):
			(last_state_size, last_batch_size, last_dtype,
				last_output) = getattr(self, "_last_zero_state")
			if (last_batch_size == batch_size and last_dtype == dtype and last_state_size == state_size):
				return last_output
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			output = rnn_cell_impl._zero_state_tensors(state_size, batch_size, dtype)
		if is_eager:
			self._last_zero_state = (state_size, batch_size, dtype, output)
		return output


class LayerRNNCell(RNNCell):
	"""Subclass of RNNCells that act like proper `tf.Layer` objects.
	For backwards compatibility purposes, most `RNNCell` instances allow their
	`call` methods to instantiate variables via `tf.get_variable`.  The underlying
	variable scope thus keeps track of any variables, and returning cached
	versions.  This is atypical of `tf.layer` objects, which separate this
	part of layer building into a `build` method that is only called once.
	Here we provide a subclass for `RNNCell` objects that act exactly as
	`Layer` objects do.  They must provide a `build` method and their
	`call` methods do not access Variables `tf.get_variable`.
	"""

	def __call__(self, inputs, state, scope=None, *args, **kwargs):
		"""Run this RNN cell on inputs, starting from the given state.
		Args:
			inputs: `2-D` tensor with shape `[batch_size, input_size]`.
			state: if `self.state_size` is an integer, this should be a `2-D Tensor`
			with shape `[batch_size, self.state_size]`.  Otherwise, if
			`self.state_size` is a tuple of integers, this should be a tuple
			with shapes `[batch_size, s] for s in self.state_size`.
			scope: optional cell scope.
			*args: Additional positional arguments.
			**kwargs: Additional keyword arguments.
		Returns:
			A pair containing:
				- Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
				- New state: Either a single `2-D` tensor, or a tuple of tensors matching
				the arity and shapes of `state`.
		"""
		# Bypass RNNCell's variable capturing semantics for LayerRNNCell.
		# Instead, it is up to subclasses to provide a proper build
		# method.  See the class docstring for more details.
		return base_layer.Layer.__call__(self, inputs, state, scope=scope, *args, **kwargs)


class ResetGRUCell(RNNCell):
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

	def __init__(
			self,
			num_units,
			t_reset=1,
			activation=None,
			reuse=None,
			kernel_initializer=None,
			bias_initializer=None,
			name=None):
		super(ResetGRUCell, self).__init__(_reuse=reuse, name=name)

		# Inputs must be 2-dimensional.
		self.input_spec = base_layer.InputSpec(ndim=3)

		self._num_units = num_units
		self._t_reset = t_reset
		self._activation = activation or math_ops.tanh
		self._kernel_initializer = kernel_initializer
		self._bias_initializer = bias_initializer

	@property
	def state_size(self):
		return tf.TensorShape([self._t_reset, self._num_units])

	@property
	def output_size(self):
		return self._num_units, tf.TensorShape([self._t_reset, self._num_units])

	def build(self, inputs_shape):
		if inputs_shape[-1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

		input_depth = inputs_shape[-1].value
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

	def call(self, inputs, state, time):
		"""Gated recurrent unit (GRU) with nunits cells."""

		state_index = tf.mod(time, self._t_reset)

		gate_inputs = tf.tensordot(array_ops.concat([inputs, state], 2), self._gate_kernel, [[-1], [0]])
		# gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 2), self._gate_kernel)
		gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

		value = math_ops.sigmoid(gate_inputs)
		r, u = array_ops.split(value=value, num_or_size_splits=2, axis=2)

		r_state = r * state

		candidate = tf.tensordot(array_ops.concat([inputs, r_state], 2), self._candidate_kernel, [[-1], [0]])
		# candidate = math_ops.matmul(array_ops.concat([inputs, r_state], 2), self._candidate_kernel)
		candidate = nn_ops.bias_add(candidate, self._candidate_bias)

		c = self._activation(candidate)
		new_h = u * state + (1 - u) * c

		new_h_current = tf.gather(new_h,state_index, axis=1)

		# here we reset the correct state
		tmp = 1-tf.scatter_nd(
			tf.expand_dims(tf.expand_dims(state_index, 0), 0), tf.constant([1.0]), tf.constant([self._t_reset]))
		reset_mask = tf.expand_dims(tf.expand_dims(tmp, 0), -1)

		new_h_reset = new_h * reset_mask

		return (new_h_current, new_h), new_h_reset


class GroupResetGRUCell(ResetGRUCell):
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

	def __init__(
			self,
			num_units,
			t_reset=1,
			group_size=1,
			activation=None,
			reuse=None,
			kernel_initializer=None,
			bias_initializer=None,
			name=None):
		super(GroupResetGRUCell, self).__init__(
			num_units,
			t_reset=t_reset,
			activation=activation,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			name=name,
			reuse=reuse)

		self._group_size = group_size
		num_replicates = float(self._t_reset) / float(self._group_size)
		self._num_replicates = int(num_replicates)

		if self._num_replicates != num_replicates:
			raise ValueError('t_reset should be a multiple of group_size')

	@property
	def state_size(self):
		return tf.TensorShape([self._num_replicates, self._num_units])

	@property
	def output_size(self):
		return self._num_units, tf.TensorShape([self._num_replicates, self._num_units])

	def call(self, inputs, state, time):
		"""Gated recurrent unit (GRU) with nunits cells."""

		state_index_in_group = tf.mod(time, self._group_size)
		group_index = tf.floor_div(time, self._group_size)
		replicate_index = tf.mod(group_index, self._num_replicates)

		gate_inputs = tf.tensordot(array_ops.concat([inputs, state], 2), self._gate_kernel, [[-1], [0]])
		# gate_inputs = math_ops.matmul(#array_ops.concat([inputs, state], 2), self._gate_kernel)
		gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

		value = math_ops.sigmoid(gate_inputs)
		r, u = array_ops.split(value=value, num_or_size_splits=2, axis=2)

		r_state = r * state

		candidate = tf.tensordot(array_ops.concat([inputs, r_state], 2), self._candidate_kernel, [[-1], [0]])
		# candidate = math_ops.matmul(array_ops.concat([inputs, r_state], 2), self._candidate_kernel)
		candidate = nn_ops.bias_add(candidate, self._candidate_bias)

		c = self._activation(candidate)
		new_h = u * state + (1 - u) * c

		new_h_current = tf.gather(new_h,replicate_index, axis=1)

		# here we reset the correct state
		tmp = 1-tf.scatter_nd(
			tf.expand_dims(tf.expand_dims(replicate_index, 0), 0),
			tf.constant([1.0]), tf.constant([self._num_replicates]))
		reset_mask = tf.expand_dims(tf.expand_dims(tmp, 0), -1)

		reset_flag = tf.equal(state_index_in_group+1, self._group_size)
		new_h_reset = tf.cond(reset_flag, lambda: new_h * reset_mask, lambda: new_h)

		return (new_h_current, new_h), new_h_reset
