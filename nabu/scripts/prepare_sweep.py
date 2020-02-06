"""@file with sweep you can try different parameter sets"""

import os
import shutil
from six.moves import configparser
import tensorflow as tf


def main(
		sweep, command, expdir, recipe, computing, resume, duplicates, duplicates_ind_offset, test_when_finished,
		allow_early_testing):
	"""main function"""
	if expdir is None:
		raise BaseException(
			'no expdir specified. Command usage: run sweep --command=your_command --expdir=/path/to/recipe '
			'--recipe=/path/to/recipe --sweep=/path/to/recipe/sweep_file')
	if recipe is None:
		raise BaseException(
			'no recipe specified. Command usage: run sweep --command=your_command --expdir=/path/to/recipe '
			'--recipe=/path/to/recipe --sweep=/path/to/recipe/sweep_file')
	if command is None:
		raise BaseException(
			'no command specified. Command usage: run sweep --command=your_command --expdir=/path/to/recipe '
			'--recipe=/path/to/recipe --sweep=/path/to/recipe/sweep_file')
	if sweep is None:
		raise BaseException(
			'no sweep specified. Command usage: run sweep --command=your_command --expdir=/path/to/recipe '
			'--recipe=/path/to/recipe --sweep=/path/to/recipe/sweep_file')

	# read the sweep file
	with open(sweep) as fid:
		sweeptext = fid.read()

	experiments = [exp.split('\n') for exp in sweeptext.strip().split('\n\n')]
	params = [[param.split() for param in exp[1:]] for exp in experiments]
	expnames = [exp[0] for exp in experiments]

	if not os.path.isdir(expdir):
		os.makedirs(expdir)

	for i, expname in enumerate(expnames):
		run_string = 'run %s --expdir=%s --recipe=%s --computing=%s' % (
				command, os.path.join(expdir, expname), os.path.join(expdir, 'recipes', expname), computing)

		if command == 'train':
			# copy the recipe dir to the expdir
			if not resume == 'True':
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

			# compose the string that will be run.
			run_string += ' --resume=%s --sweep_flag=%s --test_when_finished=%s' % (
					resume, True, test_when_finished)

		elif command == 'test':
			run_string += ' --allow_early_testing=%s' % allow_early_testing

		else:
			raise BaseException('Unknown command to sweep: %s' % command)

		if int(duplicates) > 1:
			run_string += ' --duplicates=%s --duplicates_ind_offset=%s' % (duplicates, duplicates_ind_offset)

		os.system(run_string)


if __name__ == '__main__':
	tf.app.flags.DEFINE_string('expdir', None, 'the experiments directory')
	tf.app.flags.DEFINE_string('recipe', None, 'The directory containing the recipe')
	tf.app.flags.DEFINE_string('computing', 'standard', 'the distributed computing system one of standard or condor')
	tf.app.flags.DEFINE_string('sweep', None, 'the file containing the sweep parameters')
	tf.app.flags.DEFINE_string('command', None, 'the command to run')
	tf.app.flags.DEFINE_string('resume', 'False', 'whether the experiment in expdir, if available, has to be resumed')
	tf.app.flags.DEFINE_string('duplicates', '1', 'How many duplicates of the same experiment should be run')
	tf.app.flags.DEFINE_string('duplicates_ind_offset', '0', 'Index offset for duplicates')
	tf.app.flags.DEFINE_string(
		'test_when_finished', 'False', 'whether the test script should be started upon finishing training')
	tf.app.flags.DEFINE_string(
		'allow_early_testing', 'False',
		'whether testing is allowed before training has ended (only relevant for the test command)')

	FLAGS = tf.app.flags.FLAGS

	main(
		FLAGS.sweep, FLAGS.command, FLAGS.expdir, FLAGS.recipe, FLAGS.computing, FLAGS.resume, FLAGS.duplicates,
		FLAGS.duplicates_ind_offset, FLAGS.test_when_finished, FLAGS.allow_early_testing)
