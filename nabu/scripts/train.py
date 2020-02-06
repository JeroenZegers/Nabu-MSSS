"""@file train.py
this file will do the training"""

import sys
import os
import tensorflow as tf
from six.moves import configparser
from nabu.neuralnetworks.trainers import trainer_factory
import json
import math
import warnings

sys.path.append(os.getcwd())


def train(clusterfile, job_name, task_index, ssh_command, expdir):

	""" does everything for ss training

	Args:
		clusterfile: the file where all the machines in the cluster are
			specified if None, local training will be done
		job_name: one of ps or worker in the case of distributed training
		task_index: the task index in this job
		ssh_command: the command to use for ssh, if 'None' no tunnel will be
			created
		expdir: the experiments directory
	"""

	# read the database config file
	parsed_database_cfg = configparser.ConfigParser()
	parsed_database_cfg.read(os.path.join(expdir, 'database.cfg'))

	# read the ss config file
	model_cfg = configparser.ConfigParser()
	model_cfg.read(os.path.join(expdir, 'model.cfg'))

	# read the trainer config file
	parsed_trainer_cfg = configparser.ConfigParser()
	parsed_trainer_cfg.read(os.path.join(expdir, 'trainer.cfg'))
	trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

	# read the decoder config file
	evaluator_cfg = configparser.ConfigParser()
	evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

	# read the loss config file
	losses_cfg_file = os.path.join(expdir, 'loss.cfg')
	if not os.path.isfile(losses_cfg_file):
		warnings.warn('In following versions it will be required to provide a loss config file', Warning)
		losses_cfg_available = False
		losses_cfg = None
	else:
		losses_cfg_available = True
		losses_cfg = configparser.ConfigParser()
		losses_cfg.read(losses_cfg_file)

	# Get the config files for each training stage. Each training stage has a different
	# segment length and its network is initliazed with the network of the previous
	# training stage
	segment_lengths = trainer_cfg['segment_lengths'].split(' ')
	# segment_lengths = [segment_lengths[-1]]

	val_sum = dict()
	for seg_len_ind, segment_length in enumerate(segment_lengths):
	
		segment_expdir = os.path.join(expdir, segment_length)

		segment_parsed_database_cfg = configparser.ConfigParser()
		segment_parsed_database_cfg.read(
			os.path.join(segment_expdir, 'database.cfg'))

		segment_parsed_trainer_cfg = configparser.ConfigParser()
		segment_parsed_trainer_cfg.read(
			os.path.join(segment_expdir, 'trainer.cfg'))
		segment_trainer_cfg = dict(segment_parsed_trainer_cfg.items('trainer'))

		if 'multi_task' in segment_trainer_cfg['trainer']:
			segment_tasks_cfg = dict()
			for task in segment_trainer_cfg['tasks'].split(' '):
				segment_tasks_cfg[task] = dict(segment_parsed_trainer_cfg.items(task))
		else:
			segment_tasks_cfg = None

		# If this is first segment length, and there is no previously validated training session for this segment length,
		# we can allow to use a different trained model to be used for bootstrapping the current model
		if seg_len_ind == 0 and \
			not os.path.exists(os.path.join(segment_expdir, 'logdir', 'validated.ckpt.index')) and \
			'init_file' in segment_trainer_cfg:
			if not os.path.exists(segment_trainer_cfg['init_file'] + '.index'):
				raise BaseException('The requested bootstrapping model does not exist: %s' % segment_trainer_cfg['init_file'])
			init_filename = segment_trainer_cfg['init_file']
			print('Using the following model for bootstrapping: %s' % init_filename)

		# If the above bootstrapping does not apply and there was no previously validated training sessions, use the
		# model of the previous segment length as initialization for the current one
		elif seg_len_ind > 0 and not os.path.exists(os.path.join(segment_expdir, 'logdir', 'validated.ckpt.index')):
			init_filename = os.path.join(expdir, segment_lengths[seg_len_ind-1], 'model', 'network.ckpt')
			if not os.path.exists(init_filename + '.index'):
				init_filename = None

		else:
			init_filename = None

		# if this training stage has already successfully finished, skip it
		if segment_lengths[seg_len_ind] != 'full' \
			and os.path.exists(os.path.join(expdir, segment_lengths[seg_len_ind], 'model', 'network.ckpt.index')):
			print('Already found a fully trained model for segment length %s' % segment_length)
		else:
			tr = trainer_factory.factory(segment_trainer_cfg['trainer'])(
				conf=segment_trainer_cfg,
				tasksconf=segment_tasks_cfg,
				dataconf=segment_parsed_database_cfg,
				modelconf=model_cfg,
				evaluatorconf=evaluator_cfg,
				lossesconf=losses_cfg,
				expdir=segment_expdir,
				init_filename=init_filename,
				task_index=task_index)

			print('starting training for segment length: %s' % segment_length)

			# train the model
			best_val_loss = tr.train()
			if best_val_loss is not None:
				if tr.acc_steps:
					val_sum[segment_length] = {task: round(loss*1e5)/1e5 for loss, task in zip(best_val_loss, tr.tasks)}
				else:
					val_sum[segment_length] = round(best_val_loss*1e5)/1e5

			# best_val_losses, all_tasks = tr.train()
			# if best_val_losses is not None:
			# 	val_sum[segment_length] = {task: float(loss) for (loss, task) in zip(best_val_losses, all_tasks)}

	if val_sum and 'full' in val_sum:
		out_file = os.path.join(expdir, 'val_sum.json')
		with open(out_file, 'w') as fid:
			print('the validation loss ...')
			print(val_sum)
			print('... will be saved to memory')
			json.dump(val_sum, fid)
	else:
		print('Did not find a validation loss to save')


if __name__ == '__main__':

	# define the FLAGS
	tf.app.flags.DEFINE_string('clusterfile', None, 'The file containing the cluster')
	tf.app.flags.DEFINE_string('job_name', 'local', 'One of ps, worker')
	tf.app.flags.DEFINE_integer('task_index', 0, 'The task index')
	tf.app.flags.DEFINE_string('ssh_command', 'None', 'the command that should be used to create ssh tunnels')
	tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experimental directory')

	FLAGS = tf.app.flags.FLAGS

	train(
		clusterfile=FLAGS.clusterfile,
		job_name=FLAGS.job_name,
		task_index=FLAGS.task_index,
		ssh_command=FLAGS.ssh_command,
		expdir=FLAGS.expdir)
