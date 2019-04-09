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
from nabu.neuralnetworks.trainers import task_trainer as task_trainer_script
import pdb


class MultiTaskTrainer(object):
	"""General class outlining the multi task training environment of a model."""

	def __init__(
			self,
			conf,
			tasksconf,
			dataconf,
			modelconf,
			evaluatorconf,
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

			task_trainer = task_trainer_script.TaskTrainer(
				task, conf, taskconf, self.models, modelconf, dataconf, evaluatorconf, self.batch_size)

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
				initializer=tf.constant_initializer(True),
				trainable=False)

			self.dont_save_final_model = self.should_save_final_model.assign(False).op

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
				for task_trainer in self.task_trainers:
					task_trainer.train(self.learning_rate)

				# Group ops over tasks
				self.process_minibatch = tf.group(
					*([task_trainer.process_minibatch for task_trainer in self.task_trainers]),
					name='process_minibatch_all_tasks')

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
					self.total_loss = tf.reduce_mean(self.loss_all_tasks, name='acc_loss')

				tmp = []
				for task_trainer in self.task_trainers:
					tmp.append(task_trainer.apply_gradients)

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
						self.validation_loss = tf.reduce_mean(self.val_loss_all_tasks)

					# update the learning rate factor
					self.half_lr = learning_rate_fact.assign(learning_rate_fact/2).op

					# create an operation to updated the validated step
					self.update_validated_step = validated_step.assign(self.global_step).op

					# variable to hold the best validation loss so far
					self.best_validation_all_tasks = [tf.get_variable(
						name='best_validation_task_%i' % ind,
						shape=[],
						dtype=tf.float32,
						initializer=tf.constant_initializer(1.79e+308),
						trainable=False)
						for ind in range(len(self.val_task_trainers))]

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
						trainable=False) for ind in range(len(self.val_task_trainers))]

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
						trainable=False) for ind in range(len(self.val_task_trainers))]

					# op to update the relative loss improvements
					rel_impr = [
						(self.previous_validation_all_tasks[ind]-self.val_loss_all_tasks[ind]) /
						self.previous_validation_all_tasks[ind] for ind in range(nr_tasks)]
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
						for ind in range(len(self.val_task_trainers))]

					# op to increment the number of times validation performance was worse
					self.incr_num_tries_all_tasks = [
						num_tries.assign(num_tries+1)
						for ind, num_tries in enumerate(self.num_tries_all_tasks)]

					# op to reset the number of times validation performance was worse
					self.reset_num_tries_all_tasks = [
						num_tries.assign(0)
						for ind, num_tries in enumerate(self.num_tries_all_tasks)]

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

					tf.summary.scalar('validation loss', self.validation_loss)

			else:
				self.process_val_batch = None

			tf.summary.scalar('learning rate', self.learning_rate)

			# create a histogram for all trainable parameters
			for param in tf.trainable_variables():
				tf.summary.histogram(param.name, param)

			# create the scaffold
			self.scaffold = tf.train.Scaffold()

	def train(self):
		"""train the model"""

		# start the session and standard services
		config = tf.ConfigProto(device_count={'CPU': 1})
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		# config.log_device_placement = True

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
			with tf.train.MonitoredTrainingSession(
				checkpoint_dir=os.path.join(self.expdir, 'logdir'),
				scaffold=self.scaffold,
				hooks=chief_only_hooks,
				# chief_only_hooks=chief_only_hooks,
				config=config) as sess:

				# set the number of steps
				self.set_num_steps.run(session=sess)

				# print the params that will be updated
				print 'parameters that will be trained:'
				for ind, param in enumerate(all_params):
					print 'param ind %i: %s' % (ind, param.name)
				param_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
				print 'Trainable parameter count is %d' % param_count
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
					print \
						'%d for main, %d for auxilary LSTM, %d for feedforward and output layer' %\
						(cnn_param_count, lstm_param_count, ff_param_count)
				except:
					pass

				# get the previous best validation loss for each validation task
				prev_best_val_loss_all_tasks = sess.run(self.best_validation_all_tasks)

				# start the training loop
				# pylint: disable=E1101
				while not (sess.should_stop() or self.should_stop.eval(session=sess)):
					## Validation part
					# check if validation is due
					if self.process_val_batch is not None and self.should_validate.eval(session=sess):
						if self.is_chief:
							print ('WORKER %d: validating model' % self.task_index)

							# get the previous best validation loss for each validation task
							prev_best_val_loss_all_tasks = sess.run(self.best_validation_all_tasks)

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
									'WORKER %d: validation loss:%.6g, time: %f sec' %
									(self.task_index, validation_loss, time.time()-start))
							# if multiple tasks, also print individual task losses
							if len(val_loss_all_tasks) > 1:
								for ind, loss_task in enumerate(val_loss_all_tasks):
									print_str += (
											', task_loss %s: %.6g' %
											(self.task_trainers[ind].task_name, loss_task))
							print print_str

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
									print (
											'WORKER %d: validation loss is worse for %s!' %
											(self.task_index, val_task.task_name))

									# check how many times validation performance was
									# worse
									sess.run(self.incr_num_tries_all_tasks[task_ind])
									if self.conf['num_tries'] != 'None':
										num_tries = sess.run(self.num_tries_all_tasks[task_ind])
										if num_tries == int(self.conf['num_tries']):
											terminate_train = True
									global_step = sess.run(self.global_step)
									if global_step > 1000 and np.sum(np.abs(rel_val_loss_all_tasks[task_ind])) < 0.004:
										print \
											'WORKER %d: Relative improvements are becoming to small. Terminating ' \
											'training' % self.task_index
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

							# decide what to do for training based on the above task validations
							if terminate_train:
								validation_hook.restore()
								print ('WORKER %d: terminating training' % self.task_index)
								self.terminate.run(session=sess)
								break

							if restore_validation:
								# wait untill all workers are at validation
								# point
								while not self.all_waiting.eval(
									session=sess):
									time.sleep(1)
								self.reset_waiting.run(session=sess)

								print ('WORKER %d: loading previous model' % self.task_index)

								# load the previous model
								validation_hook.restore()

							if continue_validation:
								self.update_validated_step.run(session=sess)

							if do_halve_lr:
								print ('WORKER %d: halving learning rate' % self.task_index)
								self.half_lr.run(session=sess)
								validation_hook.save()

							#
							if np.sum(sess.run(self.num_tries_all_tasks)) == 0:
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

					# First, accumulate the gradients
					for _ in range(int(self.conf['numbatches_to_aggregate'])):
						_ = sess.run([self.process_minibatch])

					# Then, normalize the gradients
					_ = sess.run([self.normalize_gradients])

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

					if any(np.isnan(loss_all_tasks)):
						print (
								'WORKER %d: terminating training because loss became "Not A Number". '
								'Best model will not be saved' % self.task_index)
						self.dont_save_final_model.run(session=sess)
						self.terminate.run(session=sess)

					# Calculate loss and step size over all task optimizations
					loss = np.mean(loss_all_tasks)
					new_param_values = new_task_param_values[-1]
					params_diff = [
						10000.0*np.mean(np.abs(new_param_values[ind]-old_param_values[ind]))
						for ind in range(len(new_param_values))]

					# _, loss,loss_all_tasks, lr, global_step, num_steps,new_param_values = sess.run(
					# fetches=[self.update_op,
					# self.total_loss,
					# self.loss_all_tasks,
					# self.learning_rate,
					# self.global_step,
					# self.num_steps,
					# all_params])

					# # Output prompt
					# Start the printing string with most important information
					print_str = (
							'WORKER %d: step %d/%d loss: %.6g, learning rate: %f, time: %.2f sec' %
							(
								self.task_index,
								global_step,
								num_steps,
								loss, lr, time.time()-start))

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

					# print the complete string
					print(print_str)

		return prev_best_val_loss_all_tasks, self.tasks