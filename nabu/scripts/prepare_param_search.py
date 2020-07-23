"""@file with hyper parameter search. You can try to find the optimal parameter set"""

import os
import shutil
import cPickle as pickle
import tensorflow as tf

from nabu.hyperparameteroptimization.hyper_param_optimizer import HyperParamOptimizer


def main(hyper_param_conf, command, expdir, recipe, computing, resume):
	"""main function"""
	if 'train' not in command:
		raise ValueError('Command should be "train" and not %s' % command)

	exp_recipe_dir = os.path.join(expdir, 'recipes')
	exp_proposal_watch_dir = os.path.join(expdir, 'proposal_watch_dir')
	# optimizer_pickle_file = os.path.join(expdir, 'optimizer_only_valid_losses.pkl')
	optimizer_pickle_file = os.path.join(expdir, 'optimizer.pkl')

	if resume:
		if not os.path.isdir(expdir) or not os.path.isdir(exp_recipe_dir) or not os.path.isfile(optimizer_pickle_file):
			raise ValueError('Cannot resume as no optimizer was found in %s' % expdir)

		with open(optimizer_pickle_file, 'r') as fid:
			optimizer = pickle.load(fid)
		optimizer.max_parallel_jobs = 3
		optimizer.num_iters = 30
		optimizer.start_new_run_flag = False
		# optimizer.resume = True

	else:
		if os.path.isdir(expdir):
			shutil.rmtree(expdir)
		os.makedirs(expdir)
		if os.path.isdir(exp_recipe_dir):
			shutil.rmtree(exp_recipe_dir)
		os.makedirs(exp_recipe_dir)
		if os.path.isdir(exp_proposal_watch_dir):
			shutil.rmtree(exp_proposal_watch_dir)
		os.makedirs(exp_proposal_watch_dir)

		optimizer = HyperParamOptimizer(
			hyper_param_conf, command, expdir, exp_recipe_dir, recipe, computing, exp_proposal_watch_dir)
		print 'The following hyper parameters will be optimized: '
		print optimizer.hyper_param_names

	optimizer()


if __name__ == '__main__':
	print 'Have to check if all flags make sense for hyper paramter search'
	tf.app.flags.DEFINE_string('expdir', None, 'the exeriments directory')
	tf.app.flags.DEFINE_string('recipe', None, 'The directory containing the recipe')
	tf.app.flags.DEFINE_string('computing', 'standard', 'the distributed computing system one of standard or condor')
	tf.app.flags.DEFINE_string(
		'hyper_param_conf', 'hyper_param_conf', 'the file containing the hyper paramaters to optimize')
	tf.app.flags.DEFINE_string('command', 'train', 'the command to run, should be train')
	tf.app.flags.DEFINE_string('resume', 'True', 'whether the optimizer in expdir has to be resumed')

	FLAGS = tf.app.flags.FLAGS

	main(
		FLAGS.hyper_param_conf, FLAGS.command, FLAGS.expdir, FLAGS.recipe, FLAGS.computing, FLAGS.resume == 'True')
