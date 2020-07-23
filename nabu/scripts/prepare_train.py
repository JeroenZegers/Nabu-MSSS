"""@file main.py
this is the file that should be run for experiments"""

import sys
import os
import shutil
import subprocess
import tensorflow as tf
from six.moves import configparser
from train import train
import warnings
from datetime import date
import time
sys.path.append(os.getcwd())


def main(
		expdir, recipe, computing, minmemory, mincudamemory, resume, duplicates, duplicates_ind_offset, sweep_flag,
		test_when_finished):
	"""main function"""

	if expdir is None:
		raise Exception(
			'no expdir specified. Command usage: run train --expdir=/path/to/recipe --recipe=/path/to/recipe')

	if recipe is None:
		raise Exception(
			'no recipe specified. Command usage: run train --expdir=/path/to/recipe --recipe=/path/to/recipe')

	if not os.path.isdir(recipe):
		raise Exception('cannot find recipe %s' % recipe)
	if computing not in ['standard', 'condor', 'condor_dag', 'torque']:
		raise Exception('unknown computing mode: %s' % computing)

	duplicates = int(duplicates)
	duplicates_ind_offset = int(duplicates_ind_offset)

	database_cfg_file = os.path.join(recipe, 'database.conf')
	model_cfg_file = os.path.join(recipe, 'model.cfg')
	trainer_cfg_file = os.path.join(recipe, 'trainer.cfg')
	evaluator_cfg_file = os.path.join(recipe, 'validation_evaluator.cfg')
	losses_cfg_file = os.path.join(recipe, 'loss.cfg')
	losses_cfg_available = True
	if not os.path.isfile(losses_cfg_file):
		warnings.warn('In following versions it will be required to provide a loss config file', Warning)
		losses_cfg_available = False
	exp_specific_computing_cfg_file = os.path.join(recipe, 'computing.cfg')
	if os.path.isfile(exp_specific_computing_cfg_file) and computing != 'standard' and (not minmemory or not mincudamemory):
		parsed_exp_specific_computing_cfg = configparser.ConfigParser()
		parsed_exp_specific_computing_cfg.read(exp_specific_computing_cfg_file)
		exp_specific_computing_cfg = dict(parsed_exp_specific_computing_cfg.items('computing'))
		if not minmemory:
			minmemory = exp_specific_computing_cfg['minmemory']
		if not mincudamemory:
			mincudamemory = exp_specific_computing_cfg['mincudamemory']

	# read the trainer config file
	parsed_trainer_cfg = configparser.ConfigParser()
	parsed_trainer_cfg.read(trainer_cfg_file)
	trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

	for dupl_ind in range(duplicates_ind_offset, duplicates+duplicates_ind_offset):
		if duplicates > 1 or duplicates_ind_offset > 0:
			expdir_run = expdir+'_dupl%i' % dupl_ind
		else:
			expdir_run = expdir

		if os.path.isdir(os.path.join(expdir_run, 'processes')):
			shutil.rmtree(os.path.join(expdir_run, 'processes'))
		os.makedirs(os.path.join(expdir_run, 'processes'))

		if resume == 'True':
			if not os.path.isdir(expdir_run):
				raise Exception(
					'cannot find %s, please set resume to False if you want to start a new training process' %
					expdir_run)
		else:
			if os.path.isdir(os.path.join(expdir_run, 'logdir')):
				shutil.rmtree(os.path.join(expdir_run, 'logdir'))
			if not os.path.isdir(expdir_run):
				os.makedirs(expdir_run)
			if os.path.isdir(os.path.join(expdir_run, 'model')):
				shutil.rmtree(os.path.join(expdir_run, 'model'))
			os.makedirs(os.path.join(expdir_run, 'model'))

			if 'segment_lengths' in trainer_cfg:
				# create a separate directory for each training stage
				segment_lengths = trainer_cfg['segment_lengths'].split(' ')
				for seg_length in segment_lengths:
					seg_expdir_run = os.path.join(expdir_run, seg_length)

					if os.path.isdir(os.path.join(seg_expdir_run, 'logdir')):
						shutil.rmtree(os.path.join(seg_expdir_run, 'logdir'))
					if not os.path.isdir(seg_expdir_run):
						os.makedirs(seg_expdir_run)
					if os.path.isdir(os.path.join(seg_expdir_run, 'model')):
						shutil.rmtree(os.path.join(seg_expdir_run, 'model'))
					os.makedirs(os.path.join(seg_expdir_run, 'model'))

			# copy the configs to the expdir_run so they can be read there and the
			# experiment information is stored

			shutil.copyfile(database_cfg_file, os.path.join(expdir_run, 'database.cfg'))
			shutil.copyfile(model_cfg_file, os.path.join(expdir_run, 'model.cfg'))
			shutil.copyfile(evaluator_cfg_file, os.path.join(expdir_run, 'evaluator.cfg'))
			shutil.copyfile(trainer_cfg_file, os.path.join(expdir_run, 'trainer.cfg'))
			if losses_cfg_available:
				shutil.copyfile(losses_cfg_file, os.path.join(expdir_run, 'loss.cfg'))

			if 'segment_lengths' in trainer_cfg:
				# create designated database and trainer config files for each training stage

				batch_size_perseg = trainer_cfg['batch_size'].split(' ')
				numbatches_to_aggregate_perseg = trainer_cfg['numbatches_to_aggregate'].split(' ')
				initial_learning_rate_perseg = trainer_cfg['initial_learning_rate'].split(' ')
				learning_rate_decay_perseg = trainer_cfg['learning_rate_decay'].split(' ')
				if len(learning_rate_decay_perseg) == 1:
					learning_rate_decay_perseg = learning_rate_decay_perseg*len(segment_lengths)

				parsed_database_cfg = configparser.ConfigParser()
				parsed_database_cfg.read(database_cfg_file)

				for i, seg_length in enumerate(segment_lengths):
					seg_expdir_run = os.path.join(expdir_run, seg_length)

					segment_parsed_trainer_cfg = configparser.ConfigParser()
					segment_parsed_trainer_cfg.read(trainer_cfg_file)
					segment_parsed_trainer_cfg.set('trainer', 'batch_size', batch_size_perseg[i])
					segment_parsed_trainer_cfg.set(
						'trainer', 'numbatches_to_aggregate', numbatches_to_aggregate_perseg[i])
					segment_parsed_trainer_cfg.set(
						'trainer', 'initial_learning_rate', initial_learning_rate_perseg[i])
					segment_parsed_trainer_cfg.set(
						'trainer', 'learning_rate_decay', learning_rate_decay_perseg[i])
					with open(os.path.join(seg_expdir_run, 'trainer.cfg'), 'w') as fid:
						segment_parsed_trainer_cfg.write(fid)

					segment_parsed_database_cfg = configparser.ConfigParser()
					segment_parsed_database_cfg.read(database_cfg_file)

					for section in segment_parsed_database_cfg.sections():
						if 'store_dir' in dict(segment_parsed_database_cfg.items(section)).keys():
							segment_parsed_database_cfg.set(
								section, 'store_dir',
								os.path.join(segment_parsed_database_cfg.get(section, 'store_dir'), seg_length))
					with open(os.path.join(seg_expdir_run, 'database.cfg'), 'w') as fid:
						segment_parsed_database_cfg.write(fid)

		# saving job command to file if the the user has made the 'train_files_to_resume' or 'train_files_to_restart' dir.
		if computing != 'standard':
			today = date.today()
			today = today.strftime("%Y%m%d")
			if os.path.isdir('train_files_to_resume'):
				to_run = "run train --expdir=%s --recipe=%s --computing=%s --resume=True" % (expdir_run, recipe, computing)
				if minmemory:
					to_run += " --minmemory=%s" % minmemory
				if mincudamemory:
					to_run += " --mincudamemory=%s" % mincudamemory
				with open(os.path.join('train_files_to_resume', 'train' + today + expdir_run.replace('/', '')), 'w') as fid:
					fid.write('%s: %s \n %s:%s' % ('file_to_check', 'None', 'to_run', to_run))

			if os.path.isdir('train_files_to_restart'):
				to_run = "run train --expdir=%s --recipe=%s --computing=%s --resume=False" % (expdir_run, recipe, computing)
				if minmemory:
					to_run += " --minmemory=%s" % minmemory
				if mincudamemory:
					to_run += " --mincudamemory=%s" % mincudamemory
				with open(os.path.join('train_files_to_restart', 'train' + today + expdir_run.replace('/', '')), 'w') as fid:
					fid.write('%s: %s \n %s:%s' % ('file_to_check', 'None', 'to_run', to_run))

		#
		computing_cfg_file = 'config/computing/%s/%s.cfg' % (computing, 'non_distributed')

		if computing == 'standard':
			# manualy set for machine
			os.environ['CUDA_VISIBLE_DEVICES'] = '1'

			train(clusterfile=None, job_name='local', task_index=0, ssh_command='None', expdir=expdir_run)

		elif computing == 'condor':
			if not minmemory or not mincudamemory:
				# read the computing config file
				parsed_computing_cfg = configparser.ConfigParser()
				parsed_computing_cfg.read(computing_cfg_file)
				computing_cfg = dict(parsed_computing_cfg.items('computing'))

				if not minmemory:
					if sweep_flag == 'True':
						minmemory = computing_cfg['minmemory_sweep']
					else:
						minmemory = computing_cfg['minmemory']
				if not mincudamemory:
					if sweep_flag == 'True':
						mincudamemory = computing_cfg['mincudamemory_sweep']
					else:
						mincudamemory = computing_cfg['mincudamemory']

			if dupl_ind > 0:
				condor_prio_additional = -1-dupl_ind
			else:
				condor_prio_additional = 0

			if sweep_flag == 'True':
				condor_prio = str(-11 + condor_prio_additional)
			else:
				condor_prio = str(-10 + condor_prio_additional)

			if not os.path.isdir(os.path.join(expdir_run, 'outputs')):
				os.makedirs(os.path.join(expdir_run, 'outputs'))

			if test_when_finished == 'True':
				# write file of what is expected after training has completed
				file_to_check = os.path.join(expdir_run, 'full', 'model', 'network.ckpt.index')
				to_run = "run test --expdir=%s --recipe=%s --computing=%s" % (expdir_run, recipe, computing)
				with open(os.path.join('files_to_run', expdir_run.replace('/', '')), 'w') as fid:
					fid.write('%s: %s \n %s:%s' % ('file_to_check', file_to_check, 'to_run', to_run))
			# print(' '.join([
			# 	'condor_submit', 'expdir=%s' % expdir_run, 'script=nabu/scripts/train.py', 'memory=%s' % minmemory,
			# 	'condor_prio=%s' % condor_prio, 'nabu/computing/condor/non_distributed.job']))
			subprocess.call([
				'condor_submit', 'expdir=%s' % expdir_run, 'script=nabu/scripts/train.py', 'memory=%s' % minmemory,
				'cudamemory=%s' % mincudamemory, 'condor_prio=%s' % condor_prio, 'nabu/computing/condor/non_distributed.job'])

		elif computing == 'condor_dag':
			if not minmemory or not mincudamemory:
				# read the computing config file
				parsed_computing_cfg = configparser.ConfigParser()
				parsed_computing_cfg.read(computing_cfg_file)
				computing_cfg = dict(parsed_computing_cfg.items('computing'))

				if not minmemory:
					if sweep_flag == 'True':
						minmemory = computing_cfg['minmemory_sweep']
					else:
						minmemory = computing_cfg['minmemory']
				if not mincudamemory:
					if sweep_flag == 'True':
						mincudamemory = computing_cfg['mincudamemory_sweep']
					else:
						mincudamemory = computing_cfg['mincudamemory']

			if sweep_flag == 'True':
				condor_prio = str(-11 + condor_prio_additional)
			else:
				condor_prio = str(-10 + condor_prio_additional)

			if not os.path.isdir(os.path.join(expdir_run, 'outputs')):
				os.makedirs(os.path.join(expdir_run, 'outputs'))

			if test_when_finished == 'True':
				# write file of what is expected after training has completed
				file_to_check = os.path.join(expdir_run, 'full', 'model', 'network.ckpt.index')
				to_run = "run test --expdir=%s --recipe=%s --computing=%s" % (expdir_run, recipe, computing)
				with open(os.path.join('files_to_run', expdir_run.replace('/', '')), 'w') as fid:
					fid.write('%s: %s \n %s:%s' % ('file_to_check', file_to_check, 'to_run', to_run))

			# dagman stuff
			dagman_files_dir = os.path.join(expdir_run, 'dagman_files')
			if not os.path.isdir(dagman_files_dir):
				shutil.copytree('nabu/computing/condor_dag', dagman_files_dir)
				with open(os.path.join(dagman_files_dir, 'non_distributed.dag'), 'a') as fid:
					fid.write('VARS  A  script="nabu/scripts/train.py" expdir="%s" memory="%s" condor_prio="%s"' % (expdir_run, minmemory, condor_prio))

			subprocess.call(['condor_submit_dag', '-usedagdir', '%s/non_distributed.dag' % dagman_files_dir])

		elif computing == 'torque':
			# read the computing config file
			# parsed_computing_cfg = configparser.ConfigParser()
			# parsed_computing_cfg.read(computing_cfg_file)
			# computing_cfg = dict(parsed_computing_cfg.items('computing'))

			if not os.path.isdir(os.path.join(expdir_run, 'outputs')):
				os.makedirs(os.path.join(expdir_run, 'outputs'))

			call_str = \
				'qsub -v expdir=%s,script=nabu/scripts/train.py -e %s/outputs/main.err -o %s/outputs/main.out ' \
				'nabu/computing/torque/non_distributed.pbs' % (expdir_run, expdir_run, expdir_run)
			# call_str = \
			# 	'export expdir=%s; export script=nabu/scripts/train.py; ' \
			# 	'qsub nabu/computing/torque/non_distributed.pbs' % expdir_run
			process = subprocess.Popen(call_str, stdout=subprocess.PIPE, shell=True)
			proc_stdout = process.communicate()[0].strip()
			print proc_stdout

		else:
			raise Exception('Unknown computing type %s' % computing)


if __name__ == '__main__':
	tf.app.flags.DEFINE_string('expdir', None, 'the exeriments directory')
	tf.app.flags.DEFINE_string('recipe', None, 'The directory containing the recipe')
	tf.app.flags.DEFINE_string('computing', 'standard', 'the distributed computing system one of condor')
	tf.app.flags.DEFINE_string('minmemory', None, 'The minimum required computing RAM in MB. (only for non-standard computing)')
	tf.app.flags.DEFINE_string('mincudamemory', None, 'The minimum required computing CUDA GPU memory in MB. (only for non-standard computing)')
	tf.app.flags.DEFINE_string('resume', 'False', 'whether the experiment in expdir, if available, has to be resumed')
	tf.app.flags.DEFINE_string('duplicates', '1', 'How many duplicates of the same experiment should be run')
	tf.app.flags.DEFINE_string('duplicates_ind_offset', '0', 'Index offset for duplicates')
	tf.app.flags.DEFINE_string('sweep_flag', 'False', 'whether the script was called from a sweep')
	tf.app.flags.DEFINE_string(
		'test_when_finished', 'True', 'whether the test script should be started upon finishing training')

	FLAGS = tf.app.flags.FLAGS
	if FLAGS.minmemory == 'None':
		FLAGS.minmemory = None
	if FLAGS.mincudamemory == 'None':
		FLAGS.mincudamemory = None
	main(
		FLAGS.expdir, FLAGS.recipe, FLAGS.computing, FLAGS.minmemory, FLAGS.mincudamemory, FLAGS.resume,
		FLAGS.duplicates, FLAGS.duplicates_ind_offset, FLAGS.sweep_flag, FLAGS.test_when_finished)
