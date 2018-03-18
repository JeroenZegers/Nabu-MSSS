'''@file test.py
this file will test the performance of a model'''

import sys
import os
sys.path.append(os.getcwd())
import cPickle as pickle
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.neuralnetworks.components.hooks import LoadAtBegin, SummaryHook
from nabu.postprocessing.reconstructors import reconstructor_factory
from nabu.postprocessing.scorers import scorer_factory
from nabu.postprocessing.postprocessors import postprocessor_factory
import json
import pdb

def test(expdir):
    '''does everything for testing'''

    #read the database config file
    database_cfg = configparser.ConfigParser()
    database_cfg.read(os.path.join(expdir, 'database.cfg'))

    #read the model config file
    model_cfg = configparser.ConfigParser()
    model_cfg.read(os.path.join(expdir, 'model.cfg'))

    #read the evaluator config file
    evaluator_cfg = configparser.ConfigParser()
    evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))
    #quick fix
    #evaluator_cfg.set('evaluator','batch_size','5')

    #read the reconstructor config file
    reconstructor_cfg = configparser.ConfigParser()
    reconstructor_cfg.read(os.path.join(expdir, 'reconstructor.cfg'))

    #read the scorer config file
    scorer_cfg = configparser.ConfigParser()
    scorer_cfg.read(os.path.join(expdir, 'scorer.cfg'))

    #read the postprocessor config file, if it exists
    try:
	postprocessor_cfg = configparser.ConfigParser()
	postprocessor_cfg.read(os.path.join(expdir, 'postprocessor.cfg'))
	if not postprocessor_cfg.sections():
	    postprocessor_cfg = None
    except:
	postprocessor_cfg = None


    if evaluator_cfg.get('evaluator','evaluator') == 'multi_task':
	tasks = evaluator_cfg.get('evaluator','tasks').split(' ')

    else:
	raise 'unkown type of evaluation %s' %evaluator_cfg.get('evaluator','evaluator')

    #evaluate each task separately
    for task in tasks:

	rec_dir = os.path.join(expdir,'reconstructions',task)

	#load the model
	with open(os.path.join(expdir, 'model', 'model.pkl'), 'rb') as fid:
	    models = pickle.load(fid)

	if os.path.isfile(os.path.join(expdir, 'loss_%s'%task)):
	    print 'already reconstructed all signals for task %s, going straight to scoring'%task
	    if evaluator_cfg.has_option(task,'requested_utts'):
		requested_utts = int(evaluator_cfg.get(task,'requested_utts'))
	    else:
		requested_utts = int(evaluator_cfg.get('evaluator','requested_utts'))
	    if evaluator_cfg.has_option(task,'batch_size'):
		batch_size = int(evaluator_cfg.get(task,'batch_size'))
	    else:
		batch_size = int(evaluator_cfg.get('evaluator','batch_size'))
	    numbatches = int(float(requested_utts)/float(batch_size))

	else:

	    print 'Evaluating task %s' %task

	    #create the evaluator
	    evaltype = evaluator_cfg.get(task, 'evaluator')
	    evaluator = evaluator_factory.factory(evaltype)(
		conf=evaluator_cfg,
		dataconf=database_cfg,
		models=models,
		task=task)

	    #create the reconstructor

	    task_reconstructor_cfg = dict(reconstructor_cfg.items(task))
	    reconstruct_type = task_reconstructor_cfg['reconstruct_type']
	    reconstructor = reconstructor_factory.factory(reconstruct_type)(
		conf=task_reconstructor_cfg,
		evalconf=evaluator_cfg,
		dataconf=database_cfg,
		rec_dir=rec_dir,
		task=task)

	    #create the graph
	    graph = tf.Graph()

	    with graph.as_default():
		#compute the loss
		batch_loss, batch_norm, numbatches, batch_outputs, batch_seq_length = evaluator.evaluate()

		#create a hook that will load the model
		load_hook = LoadAtBegin(
		    os.path.join(expdir, 'model', 'network.ckpt'),
		    models)

		#create a hook for summary writing
		summary_hook = SummaryHook(os.path.join(expdir, 'logdir'))
		config = tf.ConfigProto(device_count = {'GPU': 0})
		#start the session
		with tf.train.SingularMonitoredSession(
		    hooks=[load_hook, summary_hook],config=config) as sess:

		    loss = 0.0
		    loss_norm = 0.0

		    for batch_ind in range(0,numbatches):
			print 'evaluating batch number %d' %batch_ind

			[batch_loss_eval, batch_norm_eval, batch_outputs_eval,
			      batch_seq_length_eval] = sess.run(
			      fetches=[batch_loss, batch_norm, batch_outputs, batch_seq_length])

			loss += batch_loss_eval
			loss_norm += batch_norm_eval

			#chosing the first seq_length
			reconstructor(batch_outputs_eval, batch_seq_length_eval)

		    loss = loss/loss_norm

	    print 'task %s: loss = %0.6g' %(task, loss)

	    #write the loss to disk
	    with open(os.path.join(expdir, 'loss_%s'%task), 'w') as fid:
		fid.write(str(loss))

	#from here on there is no need for a GPU anymore ==> score script to be run separately on
	#different machine?

	task_scorer_cfg = dict(scorer_cfg.items(task))
	score_types = task_scorer_cfg['score_type'].split(' ')

	for score_type in score_types:
	    if os.path.isfile(os.path.join(expdir, 'results_%s_%s_complete.json'%(task,score_type))):
		print 'Already found a score for task %s for score type %s, skipping it.' %(task,score_type)
	    else:

		print 'Scoring task %s for score type %s' %(task,score_type)

		#create the scorer
		scorer = scorer_factory.factory(score_type)(
		    conf=task_scorer_cfg,
		    evalconf=evaluator_cfg,
		    dataconf=database_cfg,
		    rec_dir=rec_dir,
		    numbatches=numbatches,
		    task=task)

		#run the scorer
		scorer()

		with open(os.path.join(expdir, 'results_%s_%s_complete.json'%(task,score_type)), 'w') as fid:
		    json.dump(scorer.results,fid)

		result_summary = scorer.summarize()
		with open(os.path.join(expdir, 'results_%s_%s_summary.json'%(task,score_type)), 'w') as fid:
		    json.dump(result_summary,fid)

	if postprocessor_cfg != None: # && postprocessing is not done yet for this task
	    task_postprocessor_cfg = dict(postprocessor_cfg.items(task))
	    task_processor_cfg = dict(postprocessor_cfg.items('processor_'+task))
	    postprocess_types = task_postprocessor_cfg['postprocess_type'].split(' ')

	    for postprocess_type in postprocess_types:
		#create the postprocessor
		postprocessor = postprocessor_factory.factory(postprocess_type)(
		    conf=task_postprocessor_cfg,
		    proc_conf = task_processor_cfg,
		    evalconf=evaluator_cfg,
		    expdir=expdir,
		    rec_dir=rec_dir,
		    task=task)

		#run the postprocessor
		postprocessor()

		postprocessor.matlab_eng.quit()




if __name__ == '__main__':

    tf.app.flags.DEFINE_string('expdir', 'expdir',
                               'the experiments directory that was used for'
                               ' training'
                              )
    FLAGS = tf.app.flags.FLAGS

    test(FLAGS.expdir)
