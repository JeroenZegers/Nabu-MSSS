"""@file task_trainer.py
neural network trainer environment"""

import tensorflow as tf
from nabu.neuralnetworks.loss_computers import loss_computer_factory
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.processing import input_pipeline
from nabu.neuralnetworks.models import run_multi_model
import numpy as np


class TaskTrainer(object):
	"""General class on how to train for a single task."""

	def __init__(self, task_name, trainerconf, taskconf, models, modelconf, dataconf, evaluatorconf, lossconf, batch_size):
		"""
		TaskTrainer constructor, gathers the dataconfigs and sets the loss_computer and
		evaluator for this task.

		Args:
		task_name: a name for the training task
			trainerconf: the trainer config
			taskconf: the config file for each task
			models: the neural net models
			modelconf: the neural net models configuration
			dataconf: the data configuration as a ConfigParser
			evaluatorconf: the evaluator configuration for evaluating
				if None no evaluation will be done
			lossconf: the configuration of the loss function
			batch_size: the size of the batch.
		"""

		self.task_name = task_name
		self.trainerconf = trainerconf
		self.taskconf = taskconf
		self.models = models
		self.modelconf = modelconf
		self.evaluatorconf = evaluatorconf
		self.batch_size = batch_size

		# get the database configurations for all inputs, outputs, intermediate model nodes and models.
		self.output_names = taskconf['outputs'].split(' ')
		self.input_names = taskconf['inputs'].split(' ')
		self.target_names = taskconf['targets'].split(' ')
		if self.target_names == ['']:
			self.target_names = []
		self.model_nodes = taskconf['nodes'].split(' ')

		if 'linkedsets' in taskconf:
			set_names = taskconf['linkedsets'].split(' ')
			self.linkedsets = dict()
			for set_name in set_names:
				set_input_names = ['%s_%s' % (set_name, in_name) for in_name in self.input_names]
				set_target_names = ['%s_%s' % (set_name, tar_name) for tar_name in self.target_names]
				self.linkedsets[set_name] = {'inputs': set_input_names, 'targets': set_target_names}

			if 'linkedset_weighting' in taskconf:
				linkedset_weighting = np.array(map(float, taskconf['linkedset_weighting'].split(' ')))
				# the first set has the reference weight
				linkedset_weighting /= linkedset_weighting[0]
			else:
				linkedset_weighting = np.array([1.0] * len(self.linkedsets))
			self.linkedset_weighting = {set_name: weight for set_name, weight in zip(set_names, linkedset_weighting)}
		else:
			self.linkedsets = {'set0': {'inputs': self.input_names, 'targets': self.target_names}}
			self.linkedset_weighting = {'set0': 1.0}

		self.input_dataconfs = dict()
		self.target_dataconfs = dict()
		for linkedset in self.linkedsets:
			self.input_dataconfs[linkedset] = []
			for input_name in self.linkedsets[linkedset]['inputs']:
				# input config
				dataconfs_for_input = []
				sections = taskconf[input_name].split(' ')
				for section in sections:
					dataconfs_for_input.append(dict(dataconf.items(section)))
				self.input_dataconfs[linkedset].append(dataconfs_for_input)

			self.target_dataconfs[linkedset] = []
			for target_name in self.linkedsets[linkedset]['targets']:
				# target config
				dataconfs_for_target = []
				sections = taskconf[target_name].split(' ')
				for section in sections:
					dataconfs_for_target.append(dict(dataconf.items(section)))
				self.target_dataconfs[linkedset].append(dataconfs_for_target)

		self.model_links = dict()
		self.inputs_links = dict()
		self.nodes_output_names = dict()
		for node in self.model_nodes:
			self.model_links[node] = taskconf['%s_model' % node]
			self.inputs_links[node] = taskconf['%s_inputs' % node].split(' ')
			if '%s_output_names' % node in taskconf:
				self.nodes_output_names[node] = taskconf['%s_output_names' % node].split(' ')
			else:
				self.nodes_output_names[node] = node

		# create the loss computer
		if lossconf:
			loss_type = lossconf['loss_type']
		else:
			loss_type = taskconf['loss_type']

		self.loss_computer = loss_computer_factory.factory(loss_type)(lossconf, self.batch_size)

		# create valiation evaluator
		evaltype = evaluatorconf.get('evaluator', 'evaluator')
		if evaltype != 'None':
			self.evaluator = evaluator_factory.factory(evaltype)(
				conf=evaluatorconf, dataconf=dataconf, lossconf=lossconf, models=self.models, task=task_name)

	def set_dataqueues(self):
		"""sets the data queues"""

		# check if running in distributed model
		self.data_queue = dict()
		for linkedset in self.linkedsets:
			data_queue_name = 'data_queue_%s_%s' % (self.task_name, linkedset)

			data_queue_elements, _ = input_pipeline.get_filenames(
				self.input_dataconfs[linkedset] + self.target_dataconfs[linkedset])

			number_of_elements = len(data_queue_elements)
			if 'trainset_frac' in self.taskconf:
				number_of_elements = int(float(number_of_elements) * float(self.taskconf['trainset_frac']))
			print '%d utterances will be used for training' % number_of_elements

			data_queue_elements = data_queue_elements[:number_of_elements]

			# create the data queue and queue runners
			self.data_queue[linkedset] = tf.train.string_input_producer(
				string_tensor=data_queue_elements,
				shuffle=False,
				seed=None,
				capacity=self.batch_size*2,
				shared_name=data_queue_name)

			# compute the number of steps
			if int(self.trainerconf['numbatches_to_aggregate']) == 0:
				num_steps = int(self.trainerconf['num_epochs']) * len(data_queue_elements) / self.batch_size
			else:
				num_steps = int(self.trainerconf['num_epochs']) * len(data_queue_elements) / \
							(self.batch_size * int(self.trainerconf['numbatches_to_aggregate']))

			done_ops = [tf.no_op()]

		return num_steps, done_ops

	def gather_grads(self, optimizer):
		""" Gather gradients for this task"""

		with tf.variable_scope(self.task_name):

			# a variable to hold the batch loss
			self.batch_loss = tf.get_variable(
				name='batch_loss', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)

			# a variable to hold the batch loss norm
			self.batch_loss_norm = tf.get_variable(
				name='batch_loss_norm', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0),
				trainable=False)

			# normalize the loss
			with tf.variable_scope('normalize_loss'):
				self.normalized_loss = self.batch_loss / self.batch_loss_norm

			self.process_minibatch = []
			for set_ind, linkedset in enumerate(self.linkedsets):

				inputs = dict()
				seq_lengths = dict()
				targets = dict()

				# create the input pipeline
				data, seq_length = input_pipeline.input_pipeline(
					data_queue=self.data_queue[linkedset],
					batch_size=self.batch_size,
					numbuckets=int(self.trainerconf['numbuckets']),
					dataconfs=self.input_dataconfs[linkedset] + self.target_dataconfs[linkedset]
				)

				# split data into inputs and targets
				for ind, input_name in enumerate(self.input_names):
					inputs[input_name] = data[ind]
					seq_lengths[input_name] = seq_length[ind]

				for ind, target_name in enumerate(self.target_names):
					targets[target_name] = data[len(self.input_names) + ind]

				# get the logits
				logits = run_multi_model.run_multi_model(
					models=self.models,
					model_nodes=self.model_nodes,
					model_links=self.model_links,
					inputs=inputs,
					inputs_links=self.inputs_links,
					nodes_output_names=self.nodes_output_names,
					output_names=self.output_names,
					seq_lengths=seq_lengths,
					is_training=True)

				# compute the loss
				task_minibatch_loss, task_minibatch_loss_norm = self.loss_computer(targets, logits, seq_lengths)
				task_minibatch_loss *= self.linkedset_weighting[linkedset]
				task_minibatch_loss_norm *= self.linkedset_weighting[linkedset]

				task_minibatch_grads_and_vars = optimizer.compute_gradients(task_minibatch_loss)

				(task_minibatch_grads, task_vars) = zip(*task_minibatch_grads_and_vars)

				if set_ind == 0:
					# This should have already been done before the loop, but then the trainable parameters where unknown
					# gather all trainable parameters
					self.params = tf.trainable_variables()

					# a variable to hold all the gradients
					self.grads = [tf.get_variable(
						param.op.name, param.get_shape().as_list(), initializer=tf.constant_initializer(0),
						trainable=False)
						for param in self.params]

				# update the batch gradients with the minibatch gradients.
				# If a minibatchgradients is None, the loss does not depent on the specific
				# variable(s) and it will thus not be updated
				with tf.variable_scope('update_gradients_%s' % linkedset):
					update_gradients = [
						grad.assign_add(batchgrad) for batchgrad, grad in zip(task_minibatch_grads, self.grads)
						if batchgrad is not None]

				acc_loss = self.batch_loss.assign_add(task_minibatch_loss)
				acc_loss_norm = self.batch_loss_norm.assign_add(task_minibatch_loss_norm)

				# group all the operations together that need to be executed to process
				# a minibatch
				self.process_minibatch.append(tf.group(
					*(update_gradients + [acc_loss] + [acc_loss_norm]), name='update_grads_loss_norm_%s' % linkedset))

			reset_batch_loss = self.batch_loss.assign(0.0)

			reset_batch_loss_norm = self.batch_loss_norm.assign(0.0)

			reset_grad = tf.variables_initializer(self.grads)

			# normalize the gradients if requested.
			with tf.variable_scope('normalize_gradients'):
				if self.trainerconf['normalize_gradients'] == 'True':
					self.normalize_gradients = [
						grad.assign(tf.divide(grad, self.batch_loss_norm)) for grad in self.grads]
				else:
					self.normalize_gradients = [grad.assign(grad) for grad in self.grads]

			# an op to reset the grads, the loss and the loss norm
			self.reset_grad_loss_norm = tf.group(*(
				[reset_grad, reset_batch_loss, reset_batch_loss_norm]), name='reset_grad_loss_norm')

			batch_grads_and_vars = zip(self.grads, task_vars)

			with tf.variable_scope('clip'):
				clip_value = float(self.trainerconf['clip_grad_value'])
				# clip the gradients
				batch_grads_and_vars = [
					(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in batch_grads_and_vars]

		return batch_grads_and_vars

	def train(self, learning_rate):
		"""set the training ops for this task"""

		# create the optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate)

		# gather the gradients
		batch_grads_and_vars = self.gather_grads(optimizer)

		# an op to apply the accumulated gradients to the variables
		self.apply_gradients = optimizer.apply_gradients(grads_and_vars=batch_grads_and_vars, name='apply_gradients')

	def evaluate_evaluator(self):
		"""set the evaluation ops for this task"""

		with tf.variable_scope(self.task_name):
			# a variable to hold the validation loss
			loss = tf.get_variable(
					name='loss',
					shape=[],
					dtype=tf.float32,
					initializer=tf.constant_initializer(0),
					trainable=False)

			reset_loss = loss.assign(0.0)

			# a variable to hold the validation loss norm
			loss_norm = tf.get_variable(
					name='loss_norm',
					shape=[],
					dtype=tf.float32,
					initializer=tf.constant_initializer(0),
					trainable=False)

			reset_loss_norm = loss_norm.assign(0.0)

			# evaluate a validation batch
			val_batch_loss, val_batch_norm, valbatches, _, _, _ = self.evaluator.evaluate()

			acc_loss = loss.assign_add(val_batch_loss)
			acc_loss_norm = loss_norm.assign_add(val_batch_norm)

			# group all the operations together that need to be executed to process
			# a validation batch
			self.process_val_batch = tf.group(*([acc_loss, acc_loss_norm]), name='update_loss')

			# an op to reset the loss and the loss norm
			self.reset_val_loss_norm = tf.group(*([reset_loss, reset_loss_norm]), name='reset_val_loss_norm')

			# normalize the loss
			self.val_loss_normalized = loss/loss_norm

		return valbatches
