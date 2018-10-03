'''@file main.py
this is the file that should be run for experiments'''

import sys
import os
sys.path.append(os.getcwd())
import socket
import shutil
import atexit
import subprocess
from time import sleep
import copy
import tensorflow as tf
from six.moves import configparser
from nabu.computing import cluster, local_cluster
from nabu.computing.static import run_remote
from nabu.computing.static import kill_processes
from train import train
import pdb

def main(expdir, recipe, computing, resume, duplicates, sweep_flag):
    '''main function'''

    if expdir is None:
        raise Exception('no expdir specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(recipe):
        raise Exception('cannot find recipe %s' % recipe)
    if computing not in ['standard', 'condor']:
        raise Exception('unknown computing mode: %s' % computing)

    duplicates=int(duplicates)

    database_cfg_file = os.path.join(recipe, 'database.conf')
    model_cfg_file = os.path.join(recipe, 'model.cfg')
    trainer_cfg_file = os.path.join(recipe, 'trainer.cfg')
    evaluator_cfg_file = os.path.join(recipe, 'validation_evaluator.cfg')

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(trainer_cfg_file)
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))
    
    for dupl_ind in range(duplicates):
	if duplicates>1:
	    expdir_run=expdir+'_dupl%i'%(dupl_ind)
	else:
	    expdir_run=expdir

	if os.path.isdir(os.path.join(expdir_run, 'processes')):
	    shutil.rmtree(os.path.join(expdir_run, 'processes'))
	os.makedirs(os.path.join(expdir_run, 'processes'))

	if resume == 'True':
	    if not os.path.isdir(expdir_run):
		raise Exception('cannot find %s, please set resume to '
				'False if you want to start a new training process'
				% expdir_run)
	else:
	    if os.path.isdir(os.path.join(expdir_run, 'logdir')):
		shutil.rmtree(os.path.join(expdir_run, 'logdir'))
	    if not os.path.isdir(expdir_run):
		os.makedirs(expdir_run)
	    if os.path.isdir(os.path.join(expdir_run, 'model')):
		shutil.rmtree(os.path.join(expdir_run, 'model'))
	    os.makedirs(os.path.join(expdir_run, 'model'))
	    
	    if 'segment_lengths' in trainer_cfg:
		#create a separate directory for each training stage
		segment_lengths = trainer_cfg['segment_lengths'].split(' ')
		for seg_length in segment_lengths:
		    seg_expdir_run = os.path.join(expdir_run,seg_length)
		    
		    if os.path.isdir(os.path.join(seg_expdir_run, 'logdir')):
			shutil.rmtree(os.path.join(seg_expdir_run, 'logdir'))
		    if not os.path.isdir(seg_expdir_run):
			os.makedirs(seg_expdir_run)
		    if os.path.isdir(os.path.join(seg_expdir_run, 'model')):
			shutil.rmtree(os.path.join(seg_expdir_run, 'model'))
		    os.makedirs(os.path.join(seg_expdir_run, 'model'))		

	    #copy the configs to the expdir_run so they can be read there and the
	    #experiment information is stored

	    shutil.copyfile(database_cfg_file,
			    os.path.join(expdir_run, 'database.cfg'))
	    shutil.copyfile(model_cfg_file,
			    os.path.join(expdir_run, 'model.cfg'))
	    shutil.copyfile(evaluator_cfg_file,
			    os.path.join(expdir_run, 'evaluator.cfg'))
	    shutil.copyfile(trainer_cfg_file, 
			    os.path.join(expdir_run, 'trainer.cfg'))

	    if 'segment_lengths' in trainer_cfg:
		#create designated database and trainer config files for each training stage
		    
		batch_size_perseg = trainer_cfg['batch_size'].split(' ')
		numbatches_to_aggregate_perseg = trainer_cfg['numbatches_to_aggregate'].split(' ')
		initial_learning_rate_perseg = trainer_cfg['initial_learning_rate'].split(' ')
		learning_rate_decay_perseg = trainer_cfg['learning_rate_decay'].split(' ')
		if len(learning_rate_decay_perseg)==1:
		    learning_rate_decay_perseg=learning_rate_decay_perseg*len(segment_lengths)
		
		parsed_database_cfg = configparser.ConfigParser()
		parsed_database_cfg.read(database_cfg_file)
		
		for i,seg_length in enumerate(segment_lengths):
		    seg_expdir_run = os.path.join(expdir_run,seg_length)
		    
		    segment_parsed_trainer_cfg = configparser.ConfigParser()
		    segment_parsed_trainer_cfg.read(trainer_cfg_file)
		    segment_parsed_trainer_cfg.set('trainer','batch_size',batch_size_perseg[i])
		    segment_parsed_trainer_cfg.set('trainer','numbatches_to_aggregate',
				    numbatches_to_aggregate_perseg[i])
		    segment_parsed_trainer_cfg.set('trainer','initial_learning_rate',
				    initial_learning_rate_perseg[i])
		    segment_parsed_trainer_cfg.set('trainer','learning_rate_decay',
				    learning_rate_decay_perseg[i])
		    with open(os.path.join(seg_expdir_run, 'trainer.cfg'), 'w') as fid:
			segment_parsed_trainer_cfg.write(fid)
		    
		    segment_parsed_database_cfg = configparser.ConfigParser()
		    segment_parsed_database_cfg.read(database_cfg_file)
		    
		    for section in segment_parsed_database_cfg.sections():
			if 'store_dir' in dict(segment_parsed_database_cfg.items(section)).keys():
			    segment_parsed_database_cfg.set(section,'store_dir',
				    os.path.join(segment_parsed_database_cfg.get(section,'store_dir'),
				    seg_length) )
		    with open(os.path.join(seg_expdir_run, 'database.cfg'), 'w') as fid:
			segment_parsed_database_cfg.write(fid)


	computing_cfg_file = 'config/computing/%s/%s.cfg' % (computing,
							    'non_distributed')

	if computing == 'standard':

	    #manualy set for machine
	    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	    train(clusterfile=None,
		  job_name='local',
		  task_index=0,
		  ssh_command='None',
		  expdir=expdir_run)

	elif computing == 'condor':

	    if not os.path.isdir(os.path.join(expdir_run, 'outputs')):
		os.makedirs(os.path.join(expdir_run, 'outputs'))

	    #read the computing config file
	    parsed_computing_cfg = configparser.ConfigParser()
	    parsed_computing_cfg.read(computing_cfg_file)
	    computing_cfg = dict(parsed_computing_cfg.items('computing'))
		
	    if sweep_flag == 'True':
		condor_prio='-8'
		minmemory = computing_cfg['minmemory_sweep']
	    else:
		condor_prio='-4'
		minmemory = computing_cfg['minmemory']
	      

	    subprocess.call(['condor_submit', 'expdir=%s' % expdir_run,
			    'script=nabu/scripts/train.py',
			    'memory=%s' % minmemory,
			    'condor_prio=%s' % condor_prio,
			    'nabu/computing/condor/non_distributed.job'])
	else:
	    raise Exception('Unknown computing type %s' % computing)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('expdir', None,
                               'the exeriments directory'
                              )
    tf.app.flags.DEFINE_string('recipe', None,
                               'The directory containing the recipe'
                              )
    tf.app.flags.DEFINE_string('computing', 'standard',
                               'the distributed computing system one of'
                               ' condor'
                              )
    tf.app.flags.DEFINE_string('resume', 'False',
                               'wether the experiment in expdir, if available, '
                               'has to be resumed'
                               )
    tf.app.flags.DEFINE_string('duplicates', '1',
                               'How many duplicates of the same experiment should be run'
                               )
    tf.app.flags.DEFINE_string('sweep_flag', 'False',
                               'wheter the script was called from a sweep'
                               )

    FLAGS = tf.app.flags.FLAGS

    main(FLAGS.expdir, FLAGS.recipe, FLAGS.computing, FLAGS.resume, 
	 FLAGS.duplicates, FLAGS.sweep_flag)
