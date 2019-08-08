"""@file model.py
contains de Model class and IterableModel"""
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np


class Model(object):
	"""a general class for a deep learning model"""
	__metaclass__ = ABCMeta

	def __init__(self, conf, name=None):
		"""Model constructor

		Args:
			conf: The model configuration as a configparser object
		"""

		self.conf = conf

		# The number of outputs of the model is one by default.
		self.num_outputs = 1

		self.scope = tf.VariableScope(False, name or type(self).__name__)

	def __call__(self, inputs, input_seq_length, is_training):

		"""
		Add the neural net variables and operations to the graph.
		The model scope attribute reuse is initialized to False. After it has
		been called for the first time, it is set to True, so that the weights
		are shared when it is called the next time

		Args:
			inputs: the inputs to the neural network, this is a dictionary of
				[batch_size x time x ...] tensors
			input_seq_length: The sequence lengths of the input utterances, this
				is a dictionary of [batch_size] vectors
			is_training: whether or not the network is in training mode

		Returns:
			- output logits, which is a dictionary of [batch_size x time x ...]
				tensors
			- the output logits sequence lengths which is a dictionary of
				[batch_size] vectors
		"""

		# compute the output logits
		logits = self._get_outputs(
			inputs=inputs,
			input_seq_length=input_seq_length,
			is_training=is_training)

		self.scope.reuse_variables()

		return logits

	@property
	def variables(self):
		"""get a list of the models's variables"""

		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)

	@abstractmethod
	def _get_outputs(self, inputs, input_seq_length, is_training):
		"""
		Add the neural net variables and operations to the graph

		Args:
			inputs: the inputs to the neural network, this is a dictionary of
				[batch_size x time x ...] tensors
			input_seq_length: The sequence lengths of the input utterances, this
				is a dictionary of [batch_size] vectors
			is_training: whether or not the network is in training mode

		Returns:
			- output logits, which is a dictionary of [batch_size x time x ...]
				tensors
		"""


class IterableModel(Model):
	"""a general class for an iterative deep learning model"""
	__metaclass__ = ABCMeta

	def __init__(self, conf, name=None):
		"""IterableModel constructor

		Args:
			conf: The model configuration as a configparser object
		"""
		super(IterableModel, self).__init__(conf,name=name)
		self.max_iters = int(conf['max_iters']) or 100

	def __call__(self, inputs, input_seq_length, is_training):

		"""
		Add the neural net variables and operations to the graph.
		The model scope attribute reuse is initialized to False. After it has
		been called for the first time, it is set to True, so that the weights
		are shared when it is called the next time

		Args:
			inputs: the inputs to the neural network, this is a dictionary of
				[batch_size x time x ...] tensors
			input_seq_length: The sequence lengths of the input utterances, this
				is a dictionary of [batch_size] vectors
			is_training: whether or not the network is in training mode

		Returns:
			- output logits, which is a dictionary of [batch_size x time x ...]
				tensors
			- the output logits sequence lengths which is a dictionary of
				[batch_size] vectors
		"""

		# compute the output logits
		logits = self._get_iterable_outputs(
			inputs=inputs,
			input_seq_length=input_seq_length,
			is_training=is_training)

		self.scope.reuse_variables()

		return logits

	def _get_iterable_outputs(self, inputs, input_seq_length, is_training):
		"""
		Add the neural net variables and operations to the graph

		Args:
			inputs: the inputs to the neural network, this is a dictionary of
				[batch_size x time x ...] tensors
			input_seq_length: The sequence lengths of the input utterances, this
				is a dictionary of [batch_size] vectors
			is_training: whether or not the network is in training mode

		Returns:
			- output logits, which is a dictionary of [batch_size x time x ...]
				tensors
		"""

		with tf.variable_scope(self.scope):
			logits = tf.while_loop(
					cond=self.stop_condition,
					body=self._get_outputs,
					loop_vars=[self.zero_state(inputs, input_seq_length, is_training)],
					back_prop=True,
					maximum_iterations=self.max_iters
				)

		# set iteration dimension last
		output_indices = self.output_inds()
		if len(output_indices) == 1:
			outputs = logits[output_indices[0]]
		else:
			outputs = logits[output_indices]

		return outputs

	@abstractmethod
	def _get_outputs(self, inputs):
		"""
		Add the neural net variables and operations to the graph

		Args:
			inputs: see zero_state

		Returns:
			- output logits, which is a dictionary of [batch_size x time x ...]
				tensors
		"""

	def stop_condition(self, state):
		"""
		Define a stop condition to halt the iterative process

		Args:
			state: the state of the iterative process

		Returns:
			stop_cond: a boolean indicating wether to stop or not
		"""
		stop_cond = state.iter_count < self.max_iters
		return stop_cond

	@abstractmethod
	def zero_state(self, inputs, input_seq_length, is_training):

		"""
		Define a zero state to initialize the iterative process

		Args:
			inputs: the inputs to the neural network, this is a dictionary of
				[batch_size x time x ...] tensors
			input_seq_length: The sequence lengths of the input utterances, this
				is a dictionary of [batch_size] vectors
			is_training: whether or not the network is in training mode

		Returns:
			zero_state: initial state for the iterative process

		"""

	@property
	@abstractmethod
	def output_inds(self):

		"""
		Indices that indicate with loopvariables will be used in the output of the model

		Returns:
			indices: indicate with loopvariables will be used in the output of the model

		"""