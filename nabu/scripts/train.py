'''@file train.py
this file will do the training'''

import sys
import os
sys.path.append(os.getcwd())
import tensorflow as tf
from six.moves import configparser
from nabu.computing import create_server
from nabu.neuralnetworks.trainers import trainer_factory
import pdb

def train(clusterfile,
          job_name,
          task_index,
          ssh_command,
          expdir):

    ''' does everything for ss training

    Args:
        clusterfile: the file where all the machines in the cluster are
            specified if None, local training will be done
        job_name: one of ps or worker in the case of distributed training
        task_index: the task index in this job
        ssh_command: the command to use for ssh, if 'None' no tunnel will be
            created
        expdir: the experiments directory
    '''
    
    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(os.path.join(expdir, 'database.cfg'))

    #read the ss config file
    model_cfg = configparser.ConfigParser()
    model_cfg.read(os.path.join(expdir, 'model.cfg'))

    #read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(os.path.join(expdir, 'trainer.cfg'))
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    #read the decoder config file
    evaluator_cfg = configparser.ConfigParser()
    evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))
    
    #Get the config files for each training stage. Each training stage has a different 
    #segment length and its network is initliazed with the network of the previous 
    #training stage
    segment_lengths = trainer_cfg['segment_lengths'].split(' ')
    #segment_lengths = [segment_lengths[-1]]
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'  
    for i,segment_length in enumerate(segment_lengths):
    
	segment_expdir = os.path.join(expdir,segment_length)

	segment_parsed_database_cfg = configparser.ConfigParser()
	segment_parsed_database_cfg.read(
	    os.path.join(segment_expdir, 'database.cfg'))
	
	segment_parsed_trainer_cfg = configparser.ConfigParser()
	segment_parsed_trainer_cfg.read(
	    os.path.join(segment_expdir, 'trainer.cfg'))
	segment_trainer_cfg = dict(segment_parsed_trainer_cfg.items('trainer'))
	
	if segment_trainer_cfg['trainer'] == 'multi_task':
	    segment_tasks_cfg = dict()
	    for task in segment_trainer_cfg['tasks'].split(' '):
		segment_tasks_cfg[task]= dict(segment_parsed_trainer_cfg.items(task))
	else:
	    segment_tasks_cfg = None
	 
	#If there was no previously validated training sessions, use the model of the 
	#previous segment length as initialization for the current one
	if i>0 and not os.path.exists(os.path.join(segment_expdir, 'logdir', 'validated.ckpt.index')):
	    init_filename = os.path.join(expdir, segment_lengths[i-1], 'model', 'network.ckpt')
	    if not os.path.exists(init_filename + '.index'):
		init_filename = None
	    
	else:
	    init_filename = None
	
	#if this training stage has already succesfully finished, skipt it
	if os.path.exists(os.path.join(expdir, segment_lengths[i], 'model', 'network.ckpt.index')):
	    print 'Already found a fully trained model for segment length %s' %segment_length
	else:
	    
	    #create the cluster and server
	    server = create_server.create_server(
		clusterfile=clusterfile,
		job_name=job_name,
		task_index=task_index,
		expdir=expdir,
		ssh_command=ssh_command)
	
	    #parameter server
	    if job_name == 'ps':
		raise 'Parameter server is currently not implemented correctly'
		##create the parameter server
		#ps = multi_task_trainer.ParameterServer(
		    #conf=segment_trainer_cfg,
		    #tasksconf=segment_tasks_cfg,
		    #modelconf=model_cfg,
		    #dataconf=segment_parsed_database_cfg,
		    #server=server,
		    #task_index=task_index)

		#if task_index ==0:
		##let the ps wait untill all workers are finished
		    #ps.join()
		    #return
	    
	    tr = trainer_factory.factory(segment_trainer_cfg['trainer'])(
		conf=segment_trainer_cfg,
		tasksconf=segment_tasks_cfg,
		dataconf=segment_parsed_database_cfg,
		modelconf=model_cfg,
		evaluatorconf=evaluator_cfg,
		expdir=segment_expdir,
		init_filename = init_filename,
		server=server,
		task_index=task_index)

	    print 'starting training for segment length: %s' %segment_length
	    
	    #train the model
	    tr.train()
	    
if __name__ == '__main__':

    #define the FLAGS
    tf.app.flags.DEFINE_string('clusterfile', None,
                               'The file containing the cluster')
    tf.app.flags.DEFINE_string('job_name', 'local', 'One of ps, worker')
    tf.app.flags.DEFINE_integer('task_index', 0, 'The task index')
    tf.app.flags.DEFINE_string(
        'ssh_command', 'None',
        'the command that should be used to create ssh tunnels')
    tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experimental directory')

    FLAGS = tf.app.flags.FLAGS

    train(
        clusterfile=FLAGS.clusterfile,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        ssh_command=FLAGS.ssh_command,
        expdir=FLAGS.expdir)
