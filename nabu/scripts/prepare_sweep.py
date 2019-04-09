"""@file with sweep you can try different parameter sets"""

import os
import shutil
from six.moves import configparser
import tensorflow as tf


def main(sweep, command, expdir, recipe, computing, resume, duplicates):
	"""main function"""
	# read the sweep file
	with open(sweep) as fid:
		sweeptext = fid.read()

	experiments = [exp.split('\n') for exp in sweeptext.strip().split('\n\n')]
	params = [[param.split() for param in exp[1:]] for exp in experiments]
	expnames = [exp[0] for exp in experiments]

	if not os.path.isdir(expdir):
		os.makedirs(expdir)

	for i, expname in enumerate(expnames):
		# copy the recipe dir to the expdir
		if os.path.isdir(os.path.join(expdir, 'recipes', expname)):
			shutil.rmtree(os.path.join(expdir, 'recipes', expname))
		shutil.copytree(recipe, os.path.join(expdir, 'recipes', expname))

		for param in params[i]:
			# read the config
			conf = configparser.ConfigParser()
			conf.read(os.path.join(expdir, 'recipes', expname, param[0]))

			# create the new configuration
			conf.set(param[1], param[2], ' '.join(param[3:]))
			with open(os.path.join(expdir, 'recipes', expname, param[0]), 'w') as fid:
				conf.write(fid)

		# run the new recipe
		if int(duplicates) == 1:
			os.system('run %s --expdir=%s --recipe=%s --computing=%s --resume=%s --sweep_flag=%s' % (
				command,
				os.path.join(expdir, expname),
				os.path.join(expdir, 'recipes', expname),
				computing,
				resume,
				True
				))
		else:
			os.system('run %s --expdir=%s --recipe=%s --computing=%s --resume=%s --duplicates=%s --sweep_flag=%s' % (
				command,
				os.path.join(expdir, expname),
				os.path.join(expdir, 'recipes', expname),
				computing,
				resume,
				duplicates,
				True
				))


if __name__ == '__main__':
	tf.app.flags.DEFINE_string('expdir', None, 'the experiments directory')
	tf.app.flags.DEFINE_string('recipe', None, 'The directory containing the recipe')
	tf.app.flags.DEFINE_string('computing', 'standard', 'the distributed computing system one of standard or condor')
	tf.app.flags.DEFINE_string('sweep', 'sweep', 'the file containing the sweep parameters')
	tf.app.flags.DEFINE_string('command', 'train', 'the command to run')
	tf.app.flags.DEFINE_string('resume', 'False', 'whether the experiment in expdir, if available, has to be resumed')
	tf.app.flags.DEFINE_string('duplicates', '1', 'How many duplicates of the same experiment should be run')

	FLAGS = tf.app.flags.FLAGS

	main(FLAGS.sweep, FLAGS.command, FLAGS.expdir, FLAGS.recipe, FLAGS.computing, FLAGS.resume, FLAGS.duplicates)
