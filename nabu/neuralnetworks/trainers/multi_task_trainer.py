"""@file multi_task_trainer.py
neural network trainer environment"""

import os
# from abc import ABCMeta, abstractmethod, abstractproperty
import time
import cPickle as Pickle
import numpy as np
import tensorflow as tf
from nabu.neuralnetworks.models import model_factory
from nabu.neuralnetworks.components import hooks
from nabu.neuralnetworks.components import sessions as nabu_sessions
from nabu.neuralnetworks.trainers import task_trainer as task_trainer_script
from tensorflow.core.protobuf import rewriter_config_pb2


class MultiTaskTrainer(object):
	"""General class outlining the multi task training environment of a model."""

	def __init__(
			self,
			conf,
			tasksconf,
			dataconf,
			modelconf,
			evaluatorconf,
			lossesconf,
			expdir,
			init_filename,
			task_index):
		"""
		MultiTaskTrainer constructor, creates the training graph

		Args:
			conf: the trainer config
			tasksconf: the config file for each task
			dataconf: the data configuration as a ConfigParser
			modelconf: the neural net model configuration
			evaluatorconf: the evaluator configuration for evaluating
				if None no evaluation will be done
			lossesconf: the configuration of the loss functions
			expdir: directory where the summaries will be written
			init_filename: filename of the network that should be used to
			initialize the model. Put to None if no network is available/wanted.
			task_index: optional index of the worker task in the cluster
		"""

		self.expdir = expdir
		self.conf = conf
		self.tasksconf = tasksconf
		self.task_index = task_index
		self.init_filename = init_filename

		self.batch_size = int(conf['batch_size'])
		self.tasks = self.conf['tasks'].split(' ')

		self.acc_steps = 'acc_steps' in self.conf and self.conf['acc_steps'] == 'True'
		self.normalize_weights_acc_steps = \
			self.acc_steps and 'normalize_weights_acc_steps' in self.conf and \
			self.conf['normalize_weights_acc_steps'] == 'True'

		if 'task_weights' in self.conf:
			self.task_weights = map(float, self.conf['task_weights'].split(' '))
			if len(self.tasks) != len(self.task_weights):
				raise BaseException('Number of task weights must equal number of tasks. but was %d and %d' % (
					len(self.task_weights), len(self.tasks)))
		else:
			self.task_weights = [1.0] * len(self.tasks)

		# create the graph
		self.graph = tf.Graph()

		# create the model
		modelfile = os.path.join(expdir, 'model', 'model.pkl')
		model_names = modelconf.get('hyper', 'model_names').split(' ')
		self.models = dict()
		with open(modelfile, 'wb') as fid:
			for model_name in model_names:
				self.models[model_name] = model_factory.factory(
					modelconf.get(model_name, 'architecture'))(
					conf=dict(modelconf.items(model_name)),
					name=model_name)
			Pickle.dump(self.models, fid)

		evaltype = evaluatorconf.get('evaluator', 'evaluator')

		# define a trainer per traintask
		self.task_trainers = []
		for task in self.tasks:
			taskconf = self.tasksconf[task]
			if lossesconf:
				lossconf = dict(lossesconf.items(taskconf['loss_type']))
			else:
				lossconf = None
			task_trainer = task_trainer_script.TaskTrainer(
				task, conf, taskconf, self.models, modelconf, dataconf, evaluatorconf, lossconf, self.batch_size)

			self.task_trainers.append(task_trainer)
		nr_tasks = len(self.task_trainers)

		num_replicas = 1
		# device = tf.DeviceSpec(job='local')

		self.is_chief = task_index == 0

		# define the placeholders in the graph
		with self.graph.as_default():

			# create a local num_steps variable
			self.num_steps = tf.get_variable(
				name='num_steps',
				shape=[],
				dtype=tf.int32,
				initializer=tf.constant_initializer(0),
				trainable=False
			)

			# a variable to hold the amount of steps already taken
			self.global_step = tf.get_variable(
				name='global_step',
				shape=[],
				dtype=tf.int32,
				initializer=tf.constant_initializer(0),
				trainable=False)

			should_terminate = tf.get_variable(
				name='should_terminate',
				shape=[],
				dtype=tf.bool,
				initializer=tf.constant_initializer(False),
				trainable=False)

			self.should_save_final_model = tf.get_variable(
				name='should_save_final_model',
				shape=[],
				dtype=tf.bool,
				initializer=tf.constant_initializer(False),
				trainable=False)

			self.do_save_final_model = self.should_save_final_model.assign(True).op

			self.terminate = should_terminate.assign(True).op

			# create a check if training should continue
			self.should_stop = tf.logical_or(
				tf.greater_equal(self.global_step, self.num_steps),
				should_terminate)

			# with tf.device(device):
			num_steps = []
			done_ops = []

			# set the dataqueues for each trainer
			for task_trainer in self.task_trainers:

				task_num_steps, task_done_ops = task_trainer.set_dataqueues()

				num_steps.append(task_num_steps)
				done_ops += task_done_ops

			self.set_num_steps = self.num_steps.assign(min(num_steps)).op
			self.done = tf.group(*done_ops)

			# training part
			with tf.variable_scope('train'):

				# a variable to scale the learning rate (used to reduce the
				# learning rate in case validation performance drops)
				learning_rate_fact = tf.get_variable(
					name='learning_rate_fact',
					shape=[],
					initializer=tf.constant_initializer(1.0),
					trainable=False)

				# compute the learning rate with exponential decay and scale
				# with the learning rate factor
				self.learning_rate = (tf.train.exponential_decay(
					learning_rate=float(conf['initial_learning_rate']),
					global_step=self.global_step,
					decay_steps=self.num_steps,
					decay_rate=float(conf['learning_rate_decay'])) * learning_rate_fact)

				# For each task, set the task specific training ops
				if self.acc_steps:
					if self.normalize_weights_acc_steps:
						# Normalize the weights used to accumulate steps over tasks. Since it is possible that some vars
						# are not optimized by all tasks, this might cause different learning rates per var per task.
						# Find which var is optimized by which task (by using a dummy optimizer)
						vars_norm_weight = dict()
						all_task_var_names = dict()
						all_task_batch_grads_and_vars = dict()
						for task_trainer, task_weight in zip(self.task_trainers, self.task_weights):
							dummy_optimizer = optimizer = tf.train.AdamOptimizer(self.learning_rate)
							task_batch_grads_and_vars = task_trainer.gather_grads(dummy_optimizer)
							task_var_names = [task_var.name for _, task_var in task_batch_grads_and_vars]
							all_task_var_names[task_trainer.task_name] = task_var_names
							all_task_batch_grads_and_vars[task_trainer.task_name] = task_batch_grads_and_vars
							for task_var_name in task_var_names:
								if task_var_name not in vars_norm_weight:
									vars_norm_weight[task_var_name] = 0.0
								vars_norm_weight[task_var_name] += task_weight

						# for each task, find the normalized var weights
						for task_trainer, task_weight in zip(self.task_trainers, self.task_weights):
							task_var_names = all_task_var_names[task_trainer.task_name]
							task_batch_grads_and_vars = all_task_batch_grads_and_vars[task_trainer.task_name]
							task_vars_norm_weights = {
								task_var_name: task_weight/vars_norm_weight[task_var_name]
								for task_var_name in task_var_names}
							task_trainer.train(
								self.learning_rate, var_weights=task_vars_norm_weights,
								batch_grads_and_vars=task_batch_grads_and_vars)

					else:

						for task_trainer, task_weight in zip(self.task_trainers, self.task_weights):
							task_trainer.train(self.learning_rate * task_weight)

				else:
					optimizer = tf.train.AdamOptimizer(self.learning_rate)
					all_batch_grads_and_vars = []
					for task_trainer in self.task_trainers:
						all_batch_grads_and_vars.append(task_trainer.gather_grads(optimizer))
					batch_grads_and_vars_dict = dict()
					for batch_grads_and_vars, task_weight in zip(all_batch_grads_and_vars, self.task_weights):
						for grad, var in batch_grads_and_vars:
							if var in batch_grads_and_vars_dict:
								batch_grads_and_vars_dict[var] += grad * task_weight
							else:
								batch_grads_and_vars_dict[var] = grad * task_weight
					batch_grads_and_vars = zip(batch_grads_and_vars_dict.values(), batch_grads_and_vars_dict.keys())
					self.batch_grads_and_vars = batch_grads_and_vars

				# Group ops over tasks
				# self.process_minibatch = tf.group(
				# 	*([task_trainer.process_minibatch for task_trainer in self.task_trainers]),
				# 	name='process_minibatch_all_tasks')

				self.reset_grad_loss_norm = tf.group(
					*([task_trainer.reset_grad_loss_norm for task_trainer in self.task_trainers]),
					name='reset_grad_loss_norm_all_tasks')

				tmp = []
				for task_trainer in self.task_trainers:
					tmp += task_trainer.normalize_gradients
				self.normalize_gradients = tf.group(*tmp, name='normalize_gradients_all_tasks')

				# accumulate losses from tasks
				with tf.variable_scope('accumulate_losses_from_tasks'):
					self.loss_all_tasks = [task_trainer.normalized_loss for task_trainer in self.task_trainers]
					self.total_loss = tf.reduce_sum(
						[loss*weight for loss, weight in zip(self.loss_all_tasks, self.task_weights)], name='acc_loss')
				# an op to apply the gradients
				if self.acc_steps:
					tmp = []
					for task_trainer in self.task_trainers:
						tmp.append(task_trainer.apply_gradients)
				else:
					# an op to apply the accumulated gradients to the variables
					self.apply_gradients = optimizer.apply_gradients(
						grads_and_vars=self.batch_grads_and_vars, name='apply_gradients')

				# all remaining operations with the UPDATE_OPS GraphKeys
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

				# an op to increment the global step
				global_step_inc = self.global_step.assign_add(1)

				# create an operation to update the gradients, the batch_loss
				# and do all other update ops
				# self.update_op = tf.group(
				# *(tmp + update_ops + [global_step_inc]),
				# name='update')

				self.other_update_op = tf.group(
					*(update_ops + [global_step_inc]),
					name='other_update')

			if evaltype != 'None':

				# validation part
				with tf.variable_scope('validate'):

					# create a variable to save the last step where the model
					# was validated
					validated_step = tf.get_variable(
						name='validated_step',
						shape=[],
						dtype=tf.int32,
						initializer=tf.constant_initializer(-int(conf['valid_frequency'])),
						trainable=False)

					# a check if validation is due
					self.should_validate = tf.greater_equal(
						self.global_step - validated_step, int(conf['valid_frequency']))

					# For each task, if requested, set the task specific validation ops
					# The number of validation batches is the minimum number of validation
					# batches over all tasks.
					tasks_excluded_for_val = ['None']
					if evaluatorconf.has_option('evaluator', 'tasks_excluded_for_val'):
						tasks_excluded_for_val = evaluatorconf.get('evaluator', 'tasks_excluded_for_val').split(' ')
					self.val_task_trainers = [
						task_trainer for task_trainer in self.task_trainers
						if task_trainer.task_name not in tasks_excluded_for_val]
					nr_val_tasks = len(self.val_task_trainers)

					valbatches = []
					for task_trainer in self.val_task_trainers:
						valbatches.append(task_trainer.evaluate_evaluator())
						self.valbatches = min(valbatches)

					# Group ops over tasks
					self.process_val_batch = tf.group(*(
						[task_trainer.process_val_batch for task_trainer in self.val_task_trainers]))

					self.reset_val_loss_norm = tf.group(*(
						[task_trainer.reset_val_loss_norm for task_trainer in self.val_task_trainers]))

					self.val_loss_all_tasks = []
					for task_trainer in self.val_task_trainers:
						self.val_loss_all_tasks.append(task_trainer.val_loss_normalized)
					self.validation_loss = tf.reduce_sum(
						[loss*weight for loss, weight in zip(self.val_loss_all_tasks, self.task_weights)])

					# update the learning rate factor
					self.half_lr = learning_rate_fact.assign(learning_rate_fact/2).op

					# create an operation to updated the validated step
					self.update_validated_step = validated_step.assign(self.global_step).op

					if self.acc_steps:
						# variable to hold the best validation loss so far
						self.best_validation_all_tasks = [tf.get_variable(
							name='best_validation_task_%i' % ind,
							shape=[],
							dtype=tf.float32,
							initializer=tf.constant_initializer(1.79e+308),
							trainable=False)
							for ind in range(nr_val_tasks)]

						# op to update the best validation loss
						self.update_best_all_tasks = [
							best_val_task.assign(self.val_loss_all_tasks[ind])
							for ind, best_val_task in enumerate(self.best_validation_all_tasks)]

						# variable to hold the previous validation loss
						self.previous_validation_all_tasks = [tf.get_variable(
							name='previous_validation_task_%i' % ind,
							shape=[],
							dtype=tf.float32,
							initializer=tf.constant_initializer(1.79e+308),
							trainable=False) for ind in range(nr_val_tasks)]

						# op to update the previous validation loss
						self.update_prev_all_tasks = [
							prev_val_task.assign(self.val_loss_all_tasks[ind])
							for ind, prev_val_task in enumerate(self.previous_validation_all_tasks)]

						# variable to hold the last x relative loss improvements. x=num_tries
						self.rel_validation_all_tasks = [tf.get_variable(
							name='rel_validation_task_%i' % ind,
							shape=[int(self.conf['num_tries'])],
							dtype=tf.float32,
							initializer=tf.constant_initializer(1.79e+308),
							trainable=False) for ind in range(nr_val_tasks)]

						# op to update the relative loss improvements
						rel_impr = [
							(self.previous_validation_all_tasks[ind]-self.val_loss_all_tasks[ind]) /
							self.previous_validation_all_tasks[ind] for ind in range(nr_val_tasks)]
						all_rel_imprs = [
							tf.concat([rel_val_task[1:],  tf.expand_dims(rel_impr[ind], -1)], axis=0)
							for ind, rel_val_task in enumerate(self.rel_validation_all_tasks)]
						self.update_rel_all_tasks = [
							tf.assign(rel_val_task, all_rel_imprs[ind])
							for ind, rel_val_task in enumerate(self.rel_validation_all_tasks)]

						# variable to hold the number of times validation performance was worse
						self.num_tries_all_tasks = [tf.get_variable(
							name='num_tries_task_%i' % ind,
							shape=[],
							dtype=tf.int32,
							initializer=tf.constant_initializer(0),
							trainable=False)
							for ind in range(nr_val_tasks)]

						# op to increment the number of times validation performance was worse
						self.incr_num_tries_all_tasks = [
							num_tries.assign(num_tries+1)
							for ind, num_tries in enumerate(self.num_tries_all_tasks)]

						# op to reset the number of times validation performance was worse
						self.reset_num_tries_all_tasks = [
							num_tries.assign(0)
							for ind, num_tries in enumerate(self.num_tries_all_tasks)]

					else:
						# variable to hold the best validation loss so far
						self.best_validation = tf.get_variable(
							name='best_validation',
							shape=[],
							dtype=tf.float32,
							initializer=tf.constant_initializer(1.79e+308),
							trainable=False)

						# op to update the best validation loss
						self.update_best = self.best_validation.assign(self.validation_loss)

						# variable to hold the previous validation loss
						self.previous_validation = tf.get_variable(
							name='previous_validation',
							shape=[],
							dtype=tf.float32,
							initializer=tf.constant_initializer(1.79e+308),
							trainable=False)

						# op to update the previous validation loss
						self.update_prev = self.previous_validation.assign(self.validation_loss)

						# variable to hold the last x relative loss improvements. x=num_tries
						self.rel_validation = tf.get_variable(
							name='rel_validation',
							shape=[int(self.conf['num_tries'])],
							dtype=tf.float32,
							initializer=tf.constant_initializer(1.79e+308),
							trainable=False)

						# op to update the relative loss improvements
						rel_impr = (self.previous_validation-self.validation_loss)/self.previous_validation
						all_rel_imprs = tf.concat([self.rel_validation[1:],  tf.expand_dims(rel_impr, -1)], axis=0)
						self.update_rel = tf.assign(self.rel_validation, all_rel_imprs)

						# variable to hold the number of times validation performance was worse
						self.num_tries = tf.get_variable(
							name='num_tries',
							shape=[],
							dtype=tf.int32,
							initializer=tf.constant_initializer(0),
							trainable=False)

						# op to increment the number of times validation performance was worse
						self.incr_num_tries = self.num_tries.assign(self.num_tries + 1)

						# op to reset the number of times validation performance was worse
						self.reset_num_tries = self.num_tries.assign(0)

					# a variable that holds the amount of workers at the
					# validation point
					waiting_workers = tf.get_variable(
						name='waiting_workers',
						shape=[],
						dtype=tf.int32,
						initializer=tf.constant_initializer(0),
						trainable=False)

					# an operation to signal a waiting worker
					self.waiting = waiting_workers.assign_add(1).op

					# an operation to set the waiting workers to zero
					self.reset_waiting = waiting_workers.initializer

					# an operation to check if all workers are waiting
					self.all_waiting = tf.equal(waiting_workers, num_replicas-1)

					# tf.summary.scalar('validation loss', self.validation_loss)

			else:
				self.process_val_batch = None

			# tf.summary.scalar('learning rate', self.learning_rate)

			# create a histogram for all trainable parameters
			# for param in tf.trainable_variables():
			# 	tf.summary.histogram(param.name, param)

			# create the scaffold
			self.scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))

	def train(self):
		"""train the model"""

		# start the session and standard services
		config = tf.ConfigProto(device_count={'CPU': 1})
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		# config.log_device_placement = True

		# added this 2 lines to fix a "AlreadyExistsError" encountered when training
		# "config/recipes_Jeroen_Default17/DBLSTM/dialogue_sepNTMAnch_DANet_4spk".
		# See https://github.com/tensorflow/tensorflow/issues/23780
		# off = rewriter_config_pb2.RewriterConfig.OFF
		# config.graph_options.rewrite_options.arithmetic_optimization = off
		# config.graph_options.rewrite_options.memory_optimization = off

		chief_only_hooks = []

		if self.init_filename is not None:
			init_hook = hooks.LoadAtBegin(self.init_filename, self.models)
			chief_only_hooks.append(init_hook)

		# create a hook for saving the final model
		save_hook = hooks.SaveAtEnd(
			os.path.join(self.expdir, 'model', 'network.ckpt'),
			self.models,
			self.should_save_final_model)
		chief_only_hooks.append(save_hook)

		# create a hook for saving and restoring the validated model
		validation_hook = hooks.ValidationSaveHook(
			os.path.join(self.expdir, 'logdir', 'validated.ckpt'),
			self.models)
		chief_only_hooks.append(validation_hook)

		stop_hook = hooks.StopHook(self.done)
		chief_only_hooks.append(stop_hook)

		# determine all parameters
		all_params = []
		for ind, _ in enumerate(self.task_trainers):
			all_params += self.task_trainers[ind].params
		all_params = list(set(all_params))
		all_params = sorted(all_params, key=lambda par: par.name)

		with self.graph.as_default():
			# with tf.train.MonitoredTrainingSession(
			# 	checkpoint_dir=os.path.join(self.expdir, 'logdir'),
			# 	scaffold=self.scaffold,
			# 	hooks=chief_only_hooks,
			# 	chief_only_hooks=chief_only_hooks,
			# 	save_summaries_steps=None,
			# 	save_summaries_secs=None,
			# 	config=config) as sess:
			with nabu_sessions.MonitoredTrainingSession(
				checkpoint_dir=os.path.join(self.expdir, 'logdir'),
				scaffold=self.scaffold,
				hooks=chief_only_hooks,
				# chief_only_hooks=chief_only_hooks,
				save_summaries_steps=None,
				save_summaries_secs=None,
				config=config) as sess:

				# set the number of steps
				self.set_num_steps.run(session=sess)

				# print the params that will be updated
				print('parameters that will be trained:')
				for ind, param in enumerate(all_params):
					print('param ind %i: %s' % (ind, param.name))
				param_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
				print('Trainable parameter count is %d' % param_count)
				try:
					cnn_param_count = np.sum([
						np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()
						if v.name.split('/')[0] == 'main'])
					lstm_param_count = np.sum([
						np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()
						if v.name.split('/')[0] == 'aux_lstm'])
					ff_param_count = np.sum([
						np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()
						if v.name.split('/')[0] in ['feedforward', 'outlayer']])
					print(\
						'%d for main, %d for auxilary LSTM, %d for feedforward and output layer' %\
						(cnn_param_count, lstm_param_count, ff_param_count))
				except:
					pass

				# get the previous best validation loss for each validation task
				if self.acc_steps:
					prev_best_val_loss_all_tasks = sess.run(self.best_validation_all_tasks)
				else:
					prev_best_val_loss = sess.run(self.best_validation)

				# start the training loop
				# pylint: disable=E1101
				while not (sess.should_stop() or self.should_stop.eval(session=sess)):
					## Validation part
					# check if validation is due
					if self.process_val_batch is not None and self.should_validate.eval(session=sess):
						if self.is_chief:
							print('Validating model')

							# get the previous best validation loss for each validation task
							if self.acc_steps:
								prev_best_val_loss_all_tasks = sess.run(self.best_validation_all_tasks)
							else:
								prev_best_val_loss = sess.run(self.best_validation)

							# reset the validation loss
							self.reset_val_loss_norm.run(session=sess)

							# start time
							start = time.time()

							# compute the validation loss
							for _ in range(self.valbatches):
								self.process_val_batch.run(session=sess)

							# get the current validation loss
							[validation_loss, val_loss_all_tasks] = sess.run(
								[self.validation_loss, self.val_loss_all_tasks])

							print_str = (
									'validation loss:%.6g, time: %f sec' %
									(validation_loss, time.time()-start))
							# if multiple tasks, also print individual task losses
							if len(val_loss_all_tasks) > 1:
								for ind, loss_task in enumerate(val_loss_all_tasks):
									print_str += (
											', task_loss %s: %.6g' %
											(self.task_trainers[ind].task_name, loss_task))
							print(print_str)

							if self.acc_steps:
								# update the relative validation improvements
								sess.run(self.update_rel_all_tasks)
								rel_val_loss_all_tasks = sess.run(self.rel_validation_all_tasks)

								# check if the validation loss is better, for every task
								terminate_train = False
								restore_validation = False
								continue_validation = True
								do_halve_lr = False
								for task_ind, val_task in enumerate(self.val_task_trainers):
									if val_loss_all_tasks[task_ind] >= prev_best_val_loss_all_tasks[task_ind]:
										print('Validation loss is worse for %s!' % val_task.task_name)

										# check how many times validation performance was
										# worse
										sess.run(self.incr_num_tries_all_tasks[task_ind])
										if self.conf['num_tries'] != 'None':
											num_tries = sess.run(self.num_tries_all_tasks[task_ind])
											if num_tries == int(self.conf['num_tries']):
												terminate_train = True
										global_step = sess.run(self.global_step)
										if global_step > 1000 and np.sum(np.abs(rel_val_loss_all_tasks[task_ind])) < 0.004:
											print('Relative improvements are becoming to small. Terminating training.')
											terminate_train = True

										if self.conf['go_back'] == 'True':
											continue_validation = False
											restore_validation = True
										else:
											continue_validation = True

										if self.conf['valid_adapt'] == 'True':
											do_halve_lr = True

									else:
										sess.run(self.update_best_all_tasks[task_ind])
										prev_best_val_loss_all_tasks[task_ind] = val_loss_all_tasks[task_ind]
										if self.conf['reset_tries'] == 'True':
											sess.run(self.reset_num_tries_all_tasks[task_ind])

								sess.run(self.update_prev_all_tasks)
							else:
								# update the relative validation improvement
								sess.run(self.update_rel)
								rel_val_loss = sess.run(self.rel_validation)

								# check if the validation loss is better
								terminate_train = False
								restore_validation = False
								continue_validation = True
								do_halve_lr = False

								if validation_loss >= prev_best_val_loss:
									print( 'Validation loss is worse!')

									# check how many times validation performance was worse
									sess.run(self.incr_num_tries)
									if self.conf['num_tries'] != 'None':
										num_tries = sess.run(self.num_tries)
										if num_tries == int(self.conf['num_tries']):
											terminate_train = True
									global_step = sess.run(self.global_step)
									if global_step > 1000 and np.sum(np.abs(rel_val_loss)) < 0.004:
										print('Relative improvements are becoming to small. Terminating training.')
										terminate_train = True

									if self.conf['go_back'] == 'True':
										continue_validation = False
										restore_validation = True
									else:
										continue_validation = True

									if self.conf['valid_adapt'] == 'True':
										do_halve_lr = True

								else:
									sess.run(self.update_best)
									prev_best_val_loss = validation_loss
									if self.conf['reset_tries'] == 'True':
										sess.run(self.reset_num_tries)

								sess.run(self.update_prev)

							# decide what to do for training based on the above task validations
							if terminate_train:
								validation_hook.restore()
								self.do_save_final_model.run(session=sess)
								print('Terminating training')
								self.terminate.run(session=sess)
								break

							if restore_validation:
								# wait untill all workers are at validation
								# point
								while not self.all_waiting.eval(
									session=sess):
									time.sleep(1)
								self.reset_waiting.run(session=sess)

								print('Loading previous model')

								# load the previous model
								validation_hook.restore()

							if continue_validation:
								self.update_validated_step.run(session=sess)

							if do_halve_lr:
								print('Halving learning rate')
								self.half_lr.run(session=sess)
								validation_hook.save()

							#
							if self.acc_steps:
								if np.sum(sess.run(self.num_tries_all_tasks)) == 0:
									self.reset_waiting.run(session=sess)

									# store the validated model
									validation_hook.save()
							else:
								if sess.run(self.num_tries) == 0:
									self.reset_waiting.run(session=sess)

									# store the validated model
									validation_hook.save()

						else:
							if self.conf['go_back'] == 'True' and self.process_val_batch is not None:
								self.waiting.run(session=sess)
								while (
										self.should_validate.eval(session=sess) and
										not self.should_stop.eval(session=sess)):
									time.sleep(1)

								if self.should_stop.eval(session=sess):
									break

					## Training part
					# start time
					start = time.time()

					# reset the gradients for the next step
					sess.run(fetches=[self.reset_grad_loss_norm])

					old_param_values = sess.run(all_params)

					# First, accumulate the gradients as often as requested for each linkedset in each task trainer.
					for _ in range(int(self.conf['numbatches_to_aggregate'])):
						for task_trainer in self.task_trainers:
							for set_ind, linkedset in enumerate(task_trainer.linkedsets):
								_ = sess.run([task_trainer.process_minibatch[set_ind]])

					# # First, accumulate the gradients
					# for _ in range(int(self.conf['numbatches_to_aggregate'])):
					# 	_ = sess.run([self.process_minibatch])

					# Then, normalize the gradients
					_ = sess.run([self.normalize_gradients])

					if self.acc_steps:
						# Finally, apply the gradients for each task optimizer. Get the variable values before
						# and after the update, so stepsizes for each task can be displayed.
						old_task_param_values = []
						new_task_param_values = []
						task_params_diff = []
						loss_all_tasks = []

						for ind, task_trainer in enumerate(self.task_trainers):
							# get the variable values before update
							if ind == 0:
								old_task_param_values.append(old_param_values)
							else:
								old_task_param_values.append(new_task_param_values[ind-1])

							# Apply the gradients in the task optimizer and get the task loss. If it is the last
							# task, also get some other stuff
							if ind+1 < len(self.task_trainers):
								[_, task_loss] = sess.run([task_trainer.apply_gradients, task_trainer.normalized_loss])
							else:
								_, _, task_loss, lr, global_step, num_steps = \
									sess.run(fetches=[
										task_trainer.apply_gradients,
										self.other_update_op,
										task_trainer.normalized_loss,
										self.learning_rate,
										self.global_step,
										self.num_steps])
							loss_all_tasks.append(task_loss)
							# get the variable values after update
							new_task_param_values.append(sess.run(all_params))

							# Calculate the stepsize for each variable by calculating the difference between old
							# and new variable values. Average this per variable type (eg weights layer 1) and average.
							# Also multiply with 10000 (this is just for printing format purposes)
							task_params_diff.append([
								10000.0*np.mean(np.abs(new_task_param_values[ind][par_ind]-old_task_param_values[ind][par_ind]))
								for par_ind in range(len(new_task_param_values[ind]))])

						# Calculate loss and step size over all task optimizations
						loss = np.sum(loss_all_tasks)
						new_param_values = new_task_param_values[-1]
						params_diff = [
							10000.0*np.mean(np.abs(new_param_values[ind]-old_param_values[ind]))
							for ind in range(len(new_param_values))]

					else:
						grads, _, _, loss, loss_all_tasks, lr, global_step, num_steps = \
							sess.run(fetches=[
								[grad for grad, var in self.batch_grads_and_vars],
								self.apply_gradients,
								self.other_update_op,
								self.total_loss,
								[task_trainer.normalized_loss for task_trainer in self.task_trainers],
								self.learning_rate,
								self.global_step,
								self.num_steps])

					if any(np.isnan(loss_all_tasks)):
						self.terminate.run(session=sess)
						raise BaseException('Terminating training because loss became "Not A Number". Best model will not be saved.')
					## Output prompt
					# Start the printing string with most important information
					print_str = \
						'step %d/%d loss: %.6g, learning rate: %f, time: %.2f sec' % \
						(global_step, num_steps, loss, lr, time.time()-start)

					# if multiple tasks, also print individual task losses
					if len(loss_all_tasks) > 1:
						print_str += ' ('
						for ind, loss_task in enumerate(loss_all_tasks):
							print_str += ('%s: %.6g. ' % (self.task_trainers[ind].task_name, loss_task))
						print_str += ')'

					if 'print_var_updates' in self.conf and self.conf['print_var_updates'] == 'True':
						# print the average variable step size
						print_str += '\n Av param upd (*10000): %.3f' % np.mean(np.array(params_diff))
						# if multiple tasks, also print individual task average variable step size
						if len(task_params_diff) > 1:
							print_str += ' ('
							for ind, task_param_diff in enumerate(task_params_diff):
								print_str += '%s: %.3f; ' % (
									self.task_trainers[ind].task_name,
									np.mean(np.array(task_param_diff)))
							print_str += ')'

						# For each variable type (eg weights layer 1) print the average step size
						print_str += ' ('
						for par_ind, param in enumerate(all_params):
							if par_ind > 0:
								print_str += ';'
							print_str += ('%i: %.3f ' % (par_ind, params_diff[par_ind]))
							# if multiple tasks, also print for each variable type the individual task average step size
							if len(task_params_diff) > 1:
								print_str += '{'
								for ind, task_param_diff in enumerate(task_params_diff):
									if ind > 0:
										print_str += '+'
									print_str += ('%.3f' % task_param_diff[par_ind])
								print_str += '} '
						print_str += ')'

					if 'print_grads_stats' in self.conf and self.conf['print_grads_stats'] == 'True':
						print_str += get_grad_stats(grads, var_names=[var.name for _, var in self.batch_grads_and_vars])

					# print the complete string
					print(print_str)

		if self.acc_steps:
			print(prev_best_val_loss_all_tasks)
			return prev_best_val_loss_all_tasks
		else:
			print(prev_best_val_loss)
			return prev_best_val_loss
		# return prev_best_val_loss_all_tasks, self.tasks


def get_grad_stats(grads, var_names):
	print_str = ""
	abs_flat_grads = [np.reshape(np.abs(grad), -1) for grad in grads]
	all_abs_grads = np.concatenate(abs_flat_grads)

	min_all_abs_grad = np.min(all_abs_grads)
	max_all_abs_grad = np.max(all_abs_grads)
	mean_all_abs_grad = np.mean(all_abs_grads)
	std_all_abs_grad = np.std(all_abs_grads)

	print_str += \
		'\n all: min_abs=%.2E, max_abs=%.2E, mean_abs=%.2E, std_abs=%.2E \n' % \
		(min_all_abs_grad, max_all_abs_grad, mean_all_abs_grad, std_all_abs_grad)

	for grad, var_name in zip(grads, var_names):
		abs_grad = np.abs(grad)
		min_abs_grad = np.min(abs_grad)
		max_abs_grad = np.max(abs_grad)
		mean_abs_grad = np.mean(abs_grad)
		std_abs_grad = np.std(abs_grad)

		print_str += \
			'\t %s: min_abs=%.2E, max_abs=%.2E, mean_abs=%.2E, std_abs=%.2E \n' % \
			(var_name, min_abs_grad, max_abs_grad, mean_abs_grad, std_abs_grad)

	return print_str
