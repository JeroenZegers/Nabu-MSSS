"""@file test.py
this file will test the performance of a model"""

import sys
import os
import numpy as np
import cPickle as pickle
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.neuralnetworks.components.hooks import LoadAtBegin #, SummaryHook
from nabu.postprocessing.reconstructors import reconstructor_factory
from nabu.postprocessing.scorers import scorer_factory
from nabu.postprocessing.speaker_verificaion_scorers import scorer_factory as speaker_verification_scorer_factory
from nabu.postprocessing.speaker_verification_handlers import speaker_verification_handler_factory
import json
import time
import warnings
sys.path.append(os.getcwd())


def test(expdir, test_model_checkpoint, task):
	"""does everything for testing"""
	# read the database config file
	database_cfg = configparser.ConfigParser()
	database_cfg.read(os.path.join(expdir, 'database.cfg'))

	# read the model config file
	model_cfg = configparser.ConfigParser()
	model_cfg.read(os.path.join(expdir, 'model.cfg'))

	# read the evaluator config file
	evaluator_cfg = configparser.ConfigParser()
	evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

	losses_cfg_file = os.path.join(expdir, 'loss.cfg')
	if not os.path.isfile(losses_cfg_file):
		warnings.warn('In following versions it will be required to provide a loss config file', Warning)
		loss_cfg = None
	else:
		loss_cfg = configparser.ConfigParser()
		loss_cfg.read(losses_cfg_file)

	if evaluator_cfg.has_option(task, 'output_handling_type'):
		output_handling_type = evaluator_cfg.get(task, 'output_handling_type')
	else:
		output_handling_type = 'reconstructor'

	if output_handling_type == 'reconstructor':
		# read the reconstructor config file
		output_handler_cfg = configparser.ConfigParser()
		output_handler_cfg.read(os.path.join(expdir, 'reconstructor.cfg'))

		rec_dir = os.path.join(expdir, 'reconstructions', task)

		# read the scorer config file
		scorer_cfg = configparser.ConfigParser()
		scorer_cfg.read(os.path.join(expdir, 'scorer.cfg'))
	elif output_handling_type == 'speaker_verification':
		# read the speaker verification output handler config file
		output_handler_cfg = configparser.ConfigParser()
		output_handler_cfg.read(os.path.join(expdir, 'speaker_verification_handler.cfg'))

		store_dir = os.path.join(expdir, 'speaker_verification_data', task)

		# read the scorer config file
		scorer_cfg = configparser.ConfigParser()
		scorer_cfg.read(os.path.join(expdir, 'speaker_verification_scorer.cfg'))

	else:
		raise BaseException('Unknown output handling type: %s' % output_handling_type)

	# read the postprocessor config file, if it exists
	try:
		postprocessor_cfg = configparser.ConfigParser()
		postprocessor_cfg.read(os.path.join(expdir, 'postprocessor.cfg'))
		if not postprocessor_cfg.sections():
			postprocessor_cfg = None
	except:
		postprocessor_cfg = None

	# load the model
	with open(os.path.join(expdir, 'model', 'model.pkl'), 'rb') as fid:
		models = pickle.load(fid)

	if \
		'/esat/spchtemp/scratch/jzegers/Nabu-SS2.0/Default17_MERL_DANet_Drude2018_sum_task_losses_sweep' in expdir or \
		'/esat/spchtemp/scratch/jzegers/Nabu-SS2.0/Default17_MERL_DANet_Drude2018_acc_step_norm_weights_sweep' in expdir:
		models['speaker_embeddings_model'].conf['no_bias'] = 'True'
		models['outlayer'].conf['no_bias'] = 'True'
		models['id_outlayer'].conf['no_bias'] = 'True'
		with open(os.path.join(expdir, 'model', 'model.pkl'), 'wb') as fid2:
			pickle.dump(models, fid2)
	elif \
		'/esat/spchtemp/scratch/jzegers/Nabu-SS2.0/Default17_SREMix_101trspks_DANet_hamming_scipy_Drude2018' in expdir:
		models['speaker_embeddings_model'].conf['no_bias'] = 'True'
		models['outlayer'].conf['no_bias'] = 'True'
		models['id_outlayer'].conf['no_bias'] = 'False'
		with open(os.path.join(expdir, 'model', 'model.pkl'), 'wb') as fid2:
			pickle.dump(models, fid2)

	if os.path.isfile(os.path.join(expdir, 'loss_%s' % task)):
		print 'Already reconstructed all signals for task %s, going straight to scoring' % task
		if evaluator_cfg.has_option(task, 'requested_utts'):
			requested_utts = int(evaluator_cfg.get(task, 'requested_utts'))
		else:
			requested_utts = int(evaluator_cfg.get('evaluator', 'requested_utts'))
		if evaluator_cfg.has_option(task, 'batch_size'):
			batch_size = int(evaluator_cfg.get(task, 'batch_size'))
		else:
			batch_size = int(evaluator_cfg.get('evaluator', 'batch_size'))
		numbatches = int(float(requested_utts)/float(batch_size))

	else:

		print 'Evaluating task %s' % task

		# create the evaluator
		if loss_cfg:
			loss_cfg = dict(loss_cfg.items(evaluator_cfg.get(task, 'loss_type')))
		evaltype = evaluator_cfg.get(task, 'evaluator')
		evaluator = evaluator_factory.factory(evaltype)(
			conf=evaluator_cfg,
			lossconf=loss_cfg,
			dataconf=database_cfg,
			models=models,
			task=task)

		checkpoint_dir = os.path.join(expdir, 'logdir_%s' % task)

		# create the output handler
		if output_handling_type == 'reconstructor':
			# create the reconstructor

			task_output_handler_cfg = dict(output_handler_cfg.items(task))
			reconstruct_type = task_output_handler_cfg['reconstruct_type']

			# whether the targets should be used to determine the optimal speaker permutation on frame level. Should
			# only be used for analysis and not for reporting results.
			if 'optimal_frame_permutation' in task_output_handler_cfg and \
				task_output_handler_cfg['optimal_frame_permutation'] == 'True':
				optimal_frame_permutation = True
			else:
				optimal_frame_permutation = False

			output_handler = reconstructor_factory.factory(reconstruct_type)(
				conf=task_output_handler_cfg,
				evalconf=evaluator_cfg,
				dataconf=database_cfg,
				rec_dir=rec_dir,
				task=task,
				optimal_frame_permutation=optimal_frame_permutation)

			if optimal_frame_permutation:
				opt_frame_perm_op = getattr(output_handler, "reconstruct_signals_opt_frame_perm", None)
				if not callable(opt_frame_perm_op):
					raise NotImplementedError(
						'The "optimal_frame_permutation" flag was set while the function '
						'"reconstruct_signals_opt_frame_perm" is not implemented in the reconstructor')

		elif output_handling_type == 'speaker_verification':
			task_output_handler_cfg = dict(output_handler_cfg.items(task))
			speaker_verification_handler_type = task_output_handler_cfg['speaker_verification_handler_type']

			output_handler = speaker_verification_handler_factory.factory(speaker_verification_handler_type)(
				conf=task_output_handler_cfg,
				evalconf=evaluator_cfg,
				dataconf=database_cfg,
				store_dir=store_dir,
				exp_dir=expdir,
				task=task)

		else:
			raise BaseException('Unknown output handling type: %s' % output_handling_type)

		# create the graph
		with tf.Graph().as_default():

			# create a hook that will load the model
			load_hook = LoadAtBegin(test_model_checkpoint, models)

			# create a hook for summary writing
			# summary_hook = SummaryHook(os.path.join(expdir, 'logdir'))

			#
			saver_hook = tf.train.CheckpointSaverHook(
				checkpoint_dir=checkpoint_dir, save_steps=np.ceil(1000.0/float(evaluator.batch_size)))

			config = tf.ConfigProto(intra_op_parallelism_threads=6,	inter_op_parallelism_threads=2,	device_count={'CPU': 8, 'GPU': 0})

			options = tf.RunOptions()
			options.report_tensor_allocations_upon_oom = True

			#
			current_batch_ind_tf = tf.get_variable(
				name='global_step',
				shape=[],
				dtype=tf.int32,
				initializer=tf.constant_initializer(0),
				trainable=False)
			current_batch_ind_inc_op = current_batch_ind_tf.assign_add(1)
			reset_current_batch_ind_op = current_batch_ind_tf.assign(0)

			# get the current batch_ind
			with tf.train.SingularMonitoredSession(config=config, checkpoint_dir=checkpoint_dir) as sess:
				start_batch_ind = sess.run(current_batch_ind_tf)
				start_utt_ind = start_batch_ind * evaluator.batch_size
				output_handler.pos = start_utt_ind

			output_handler.open_scp_files(from_start=start_utt_ind == 0)

			# compute the loss
			batch_loss, batch_norm, numbatches, batch_outputs, batch_targets, batch_seq_length = evaluator.evaluate(start_utt_ind=start_utt_ind)

			# only keep the outputs requested by the reconstructor (usually the output of the output layer)
			batch_outputs = {
				out_name: out for out_name, out in batch_outputs.iteritems()
				if out_name in output_handler.requested_output_names}
			batch_seq_length = {
				seq_name: seq for seq_name, seq in batch_seq_length.iteritems()
				if seq_name in output_handler.requested_output_names}

			hooks = [load_hook]
			# hooks = [load_hook, summary_hook]
			if numbatches > 100:
				hooks.append(saver_hook)

			# start the session
			with tf.train.SingularMonitoredSession(
				hooks=hooks, config=config, checkpoint_dir=checkpoint_dir) as sess:

				loss = 0.0
				loss_norm = 0.0

				for batch_ind in range(start_batch_ind, numbatches):
					print('evaluating batch number %d' % batch_ind)

					last_time = time.time()
					[batch_loss_eval, batch_norm_eval, batch_outputs_eval, batch_targets_eval,
						batch_seq_length_eval] = sess.run(
						fetches=[batch_loss, batch_norm, batch_outputs, batch_targets, batch_seq_length],
						options=options)

					loss += batch_loss_eval
					loss_norm += batch_norm_eval
					print('%f' % (time.time()-last_time))
					last_time = time.time()

					if output_handling_type != 'reconstructor' or not optimal_frame_permutation:
						output_handler(batch_outputs_eval, batch_seq_length_eval)
					else:
						output_handler.opt_frame_perm(batch_outputs_eval, batch_targets_eval, batch_seq_length_eval)

					sess.run(current_batch_ind_inc_op)

					print('%f' % (time.time()-last_time))

				loss = loss/loss_norm

		print('task %s: loss = %0.6g' % (task, loss))

		# write the loss to disk
		with open(os.path.join(expdir, 'loss_%s' % task), 'w') as fid:
			fid.write(str(loss))

		if hasattr(output_handler, 'scp_file'):
			output_handler.scp_fid.close()
		if hasattr(output_handler, 'masks_pointer_file'):
			output_handler.masks_pointer_fid.close()

		if os.path.isdir(checkpoint_dir):
			try:
				os.rmdir(checkpoint_dir)
			except:
				pass

	# from here on there is no need for a GPU anymore ==> score script to be run separately on
	# different machine?
	if evaluator_cfg.has_option(task, 'scorers_names'):
		scorers_names = evaluator_cfg.get(task, 'scorers_names').split(' ')
	else:
		scorers_names = [task]

	for scorer_name in scorers_names:
		task_scorer_cfg = dict(scorer_cfg.items(scorer_name))
		score_types = task_scorer_cfg['score_type'].split(' ')

		for score_type in score_types:
			if os.path.isfile(os.path.join(expdir, 'results_%s_%s_complete.json' % (scorer_name, score_type))):
				print('Already found a score for score task %s for score type %s, skipping it.' % (scorer_name, score_type))
			else:
				print('Scoring task %s for score type %s' % (scorer_name, score_type))
				checkpoint_file = os.path.join(expdir, 'checkpoint_results_%s_%s' % (scorer_name, score_type))
				if output_handling_type == 'reconstructor':
					# create the scorer
					scorer = scorer_factory.factory(score_type)(
						conf=task_scorer_cfg,
						evalconf=evaluator_cfg,
						dataconf=database_cfg,
						rec_dir=rec_dir,
						numbatches=numbatches,
						task=task,
						scorer_name=scorer_name,
						checkpoint_file=checkpoint_file)
				elif output_handling_type == 'speaker_verification':
					# create the scorer
					scorer = speaker_verification_scorer_factory.factory(score_type)(
						conf=task_scorer_cfg,
						evalconf=evaluator_cfg,
						dataconf=database_cfg,
						store_dir=store_dir,
						numbatches=numbatches,
						task=task,
						scorer_name=scorer_name,
						checkpoint_file=checkpoint_file)

				# run the scorer
				scorer()

				result_summary = scorer.summarize()

				with open(os.path.join(expdir, 'results_%s_%s_summary.json' % (scorer_name, score_type)), 'w') as fid:
					json.dump(result_summary, fid)

				with open(os.path.join(expdir, 'results_%s_%s_complete.json' % (scorer_name, score_type)), 'w') as fid:
					json.dump(scorer.storable_result(), fid)

				if os.path.isfile(checkpoint_file):
					try:
						os.remove(checkpoint_file)
					except:
						pass

	# legacy code to be removed
	if postprocessor_cfg != None:  # && postprocessing is not done yet for this task
		from nabu.postprocessing.postprocessors import postprocessor_factory

		if evaluator_cfg.has_option(task, 'postprocessors_names'):
			postprocessors_names = evaluator_cfg.get(task, 'postprocessors_names').split(' ')
		else:
			postprocessors_names = [task]

		for postprocessors_name in postprocessors_names:
			task_postprocessor_cfg = dict(postprocessor_cfg.items(postprocessors_name))
			postprocess_types = task_postprocessor_cfg['postprocess_type'].split(' ')

			for postprocess_type in postprocess_types:
				print('Postprocessing task %s for postprocessor type %s' % (postprocessors_name, postprocess_type))

				# create the postprocessor
				postprocessor = postprocessor_factory.factory(postprocess_type)(
					conf=task_postprocessor_cfg,
					evalconf=evaluator_cfg,
					expdir=expdir,
					rec_dir=rec_dir,
					postprocessors_name=postprocessors_name)

				# run the postprocessor
				postprocessor()

				postprocessor.matlab_eng.quit()


if __name__ == '__main__':

	tf.app.flags.DEFINE_string('expdir', 'expdir', 'the experiments directory that was used for training')
	tf.app.flags.DEFINE_string('test_model_checkpoint', 'test_model_checkpoint', 'the checkpointed model that will be tested')
	tf.app.flags.DEFINE_string('task', 'task', 'the name of the task to evaluate')
	FLAGS = tf.app.flags.FLAGS

	test(FLAGS.expdir, FLAGS.test_model_checkpoint, FLAGS.task)
