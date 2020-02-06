"""@file test.py
this file will run the test script"""

import sys
import os
import shutil
import subprocess
from six.moves import configparser
import tensorflow as tf
from test import test
import warnings
sys.path.append(os.getcwd())


tf.app.flags.DEFINE_string('expdir', None, 'the exeriments directory that was used for training')
tf.app.flags.DEFINE_string('recipe', None, 'The directory containing the recipe')
tf.app.flags.DEFINE_string('computing', 'standard', 'the distributed computing system one of standart or condor')
tf.app.flags.DEFINE_string('sweep_flag', 'False', 'whether the script was called from a sweep')
tf.app.flags.DEFINE_string('allow_early_testing', 'False', 'whether testing is allowed before training has ended')
tf.app.flags.DEFINE_string('duplicates', '1', 'How many duplicates of the same experiment were run')
tf.app.flags.DEFINE_string('duplicates_ind_offset', '0', 'Index offset for duplicates')

FLAGS = tf.app.flags.FLAGS


def main(_):
	"""main"""

	if FLAGS.expdir is None:
		raise Exception('no expdir specified. Command usage: run test --expdir=/path/to/recipe --recipe=/path/to/recipe')

	if FLAGS.recipe is None:
		raise Exception('no recipe specified. Command usage: run test --expdir=/path/to/recipe --recipe=/path/to/recipe')

	if not os.path.isdir(FLAGS.recipe):
		raise Exception('cannot find recipe %s' % FLAGS.recipe)

	evaluator_cfg_file = os.path.join(FLAGS.recipe, 'test_evaluator.cfg')
	database_cfg_file = os.path.join(FLAGS.recipe, 'database.conf')
	reconstructor_cfg_file = os.path.join(FLAGS.recipe, 'reconstructor.cfg')
	scorer_cfg_file = os.path.join(FLAGS.recipe, 'scorer.cfg')
	postprocessor_cfg_file = os.path.join(FLAGS.recipe, 'postprocessor.cfg')
	model_cfg_file = os.path.join(FLAGS.recipe, 'model.cfg')
	losses_cfg_file = os.path.join(FLAGS.recipe, 'loss.cfg')
	losses_cfg_available = True
	if not os.path.isfile(losses_cfg_file):
		warnings.warn('In following versions it will be required to provide a loss config file', Warning)
		losses_cfg_available = False

	# Assuming only one (typically the last one) training stage needs testing
	parsed_evaluator_cfg = configparser.ConfigParser()
	parsed_evaluator_cfg.read(evaluator_cfg_file)
	training_stage = parsed_evaluator_cfg.get('evaluator', 'segment_length')

	duplicates = int(FLAGS.duplicates)
	duplicates_ind_offset = int(FLAGS.duplicates_ind_offset)
	for dupl_ind in range(duplicates_ind_offset, duplicates+duplicates_ind_offset):
		if duplicates > 1 or duplicates_ind_offset > 1:
			expdir_run = FLAGS.expdir+'_dupl%i' % dupl_ind
		else:
			expdir_run = FLAGS.expdir

		if not os.path.isdir(expdir_run):
			raise BaseException('cannot find expdir %s' % expdir_run)

		# Find a trained model to test
		train_model_dir = os.path.join(expdir_run, training_stage, 'model')
		trained_model_checkpoint = os.path.join(train_model_dir, 'network.ckpt')
		if os.path.isfile(trained_model_checkpoint + '.meta'):
			test_expdir_run = os.path.join(expdir_run, 'test')
			test_model_dir = os.path.join(test_expdir_run, 'model')
			test_model_checkpoint = os.path.join(test_model_dir, 'network.ckpt')
		elif FLAGS.allow_early_testing == 'True':
			# Find a trained model that has been validated for the requested segment length, but has not yet finished
			train_model_dir = os.path.join(expdir_run, training_stage, 'logdir')
			trained_model_checkpoint = os.path.join(train_model_dir, 'validated.ckpt')
			if not os.path.isfile(trained_model_checkpoint + '.meta'):
				raise BaseException(
					'Early testing not allowed for %s as no model has yet been validated' %
					os.path.join(expdir_run, training_stage))
			# copy the model pickle object
			if not os.path.islink(os.path.join(train_model_dir, 'model.pkl')):
				os.symlink(
					os.path.join(expdir_run, training_stage, 'model', 'model.pkl'),
					os.path.join(train_model_dir, 'model.pkl'))
			val_step = get_early_stop_val_step(trained_model_checkpoint)
			test_expdir_run = os.path.join(expdir_run, 'test_early_%d' % val_step)
			test_model_dir = os.path.join(test_expdir_run, 'model')
			test_model_checkpoint = os.path.join(test_model_dir, 'validated.ckpt')
		else:
			exception_string = \
				'Testing not (yet) allowed for %s as training has not yet been finished and early testing is set to %s.' % \
				(os.path.join(expdir_run, training_stage), FLAGS.allow_early_testing)
			if os.path.isfile(os.path.join(expdir_run, training_stage, 'logdir', 'validated.ckpt.meta')):
				exception_string += \
					'(However, early testing would be an option as a validated, but not yet finished, model has been found)'
			raise BaseException(exception_string)
	
		# create the testing dir
		if not os.path.isdir(test_expdir_run):
			os.makedirs(test_expdir_run)
	
		# create a link to the model that will be used for testing.
		if not os.path.isdir(test_model_dir):
			os.symlink(train_model_dir, test_model_dir)
	
		# copy the config files
		parsed_database_cfg = configparser.ConfigParser()
		parsed_database_cfg.read(database_cfg_file)
		segment_parsed_database_cfg = parsed_database_cfg
	
		for section in segment_parsed_database_cfg.sections():
			if 'store_dir' in dict(segment_parsed_database_cfg.items(section)).keys():
				segment_parsed_database_cfg.set(
					section, 'store_dir', os.path.join(segment_parsed_database_cfg.get(section, 'store_dir'), training_stage))
		with open(os.path.join(test_expdir_run, 'database.cfg'), 'w') as fid:
			segment_parsed_database_cfg.write(fid)
	
		# shutil.copyfile(database_cfg_file,
						# os.path.join(test_expdir_run, 'database.cfg'))
		shutil.copyfile(evaluator_cfg_file,
						os.path.join(test_expdir_run, 'evaluator.cfg'))
		shutil.copyfile(reconstructor_cfg_file,
						os.path.join(test_expdir_run, 'reconstructor.cfg'))
		shutil.copyfile(scorer_cfg_file,
						os.path.join(test_expdir_run, 'scorer.cfg'))
	
		try:
			shutil.copyfile(postprocessor_cfg_file,
							os.path.join(test_expdir_run, 'postprocessor.cfg'))
		except:
			pass
		shutil.copyfile(model_cfg_file,
						os.path.join(test_expdir_run, 'model.cfg'))
		if losses_cfg_available:
			shutil.copyfile(losses_cfg_file,
							os.path.join(test_expdir_run, 'loss.cfg'))
	
		# Get all tasks
		evaluator_cfg = configparser.ConfigParser()
		evaluator_cfg.read(os.path.join(test_expdir_run, 'evaluator.cfg'))
		if evaluator_cfg.get('evaluator', 'evaluator') == 'multi_task':
			tasks = evaluator_cfg.get('evaluator', 'tasks').split(' ')
		else:
			raise Exception('unkown type of evaluation %s' % evaluator_cfg.get('evaluator', 'evaluator'))
	
		if FLAGS.computing == 'condor':
	
			computing_cfg_file = 'config/computing/condor/non_distributed.cfg'
			parsed_computing_cfg = configparser.ConfigParser()
			parsed_computing_cfg.read(computing_cfg_file)
			computing_cfg = dict(parsed_computing_cfg.items('computing'))
	
			if not os.path.isdir(os.path.join(test_expdir_run, 'outputs')):
				os.makedirs(os.path.join(test_expdir_run, 'outputs'))
	
			# for each task, launch a test job
			for task in tasks:
				# minmemory = computing_cfg['minmemory']
				# subprocess.call([
				#     'condor_submit', 'expdir=%s/test' % expdir_run, 'script=nabu/scripts/test.py', 'memory=%s' % minmemory,
				#                      'condor_prio=%s' % -10, 'nabu/computing/condor/non_distributed.job'])
	
				subprocess.call([
					'condor_submit', 'expdir=%s' % test_expdir_run, 'test_model_checkpoint=%s' % test_model_checkpoint,
					'task=%s' % task,
					'script=nabu/scripts/test.py', 'nabu/computing/condor/non_distributed_cpu.job'])
	
		elif FLAGS.computing == 'torque':
	
			computing_cfg_file = 'config/computing/torque/non_distributed.cfg'
			parsed_computing_cfg = configparser.ConfigParser()
			parsed_computing_cfg.read(computing_cfg_file)
			computing_cfg = dict(parsed_computing_cfg.items('computing'))
	
			if not os.path.isdir(os.path.join(test_expdir_run, 'outputs')):
				os.makedirs(os.path.join(test_expdir_run, 'outputs'))
	
			# for each task, launch a test job
			for task in tasks:
				# minmemory = computing_cfg['minmemory']
				call_str = \
					'qsub -v expdir=%s test_model_checkpoint=%s task=%s,script=nabu/scripts/test.py ' \
					'-e %s/test/outputs/main_%s.err -o %s/test/outputs/main_%s.out  nabu/computing/torque/%s' % \
					(
						test_expdir_run, test_model_checkpoint, task, test_expdir_run, task, test_expdir_run, task,
						'non_distributed_short.pbs')
				process = subprocess.Popen(call_str, stdout=subprocess.PIPE, shell=True)
				proc_stdout = process.communicate()[0].strip()
				print proc_stdout
	
		elif FLAGS.computing == 'standard':
			os.environ['CUDA_VISIBLE_DEVICES'] = '2'
			# for each task, launch a test job
			for task in tasks:
				test(expdir=test_expdir_run, test_model_checkpoint=test_model_checkpoint, task=task)
	
		else:
			raise Exception('Unknown computing type %s' % FLAGS.computing)


def get_early_stop_val_step(test_model_checkpoint):
	class tmphook(tf.train.SessionRunHook):
		def begin(self):
			collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='validate')
			self._saver = tf.train.Saver(collection, sharded=True, name='LoaderAtBegin2')

		def after_create_session(self, session, _):
			self._saver.restore(session, test_model_checkpoint)

	graph = tf.Graph()
	with graph.as_default():
		with tf.variable_scope('validate', reuse=False):
			val_step = tf.get_variable(name='validated_step', shape=[], dtype=tf.int32)


		jer = tmphook()
		with tf.train.SingularMonitoredSession(hooks=[jer]) as sess:
			val_step_value = sess.run(val_step)

	return val_step_value


if __name__ == '__main__':
	tf.app.run()
