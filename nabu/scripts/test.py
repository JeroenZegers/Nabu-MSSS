'''@file test.py
this file will test the performance of a model'''

import sys
import os
sys.path.append(os.getcwd())
import cPickle as pickle
from six.moves import configparser
import matlab.engine
import matlab
import tensorflow as tf
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.neuralnetworks.components.hooks import LoadAtBegin, SummaryHook
from nabu.postprocessing.reconstructors import reconstructor_factory
from nabu.postprocessing.scorers import scorer_factory
import json
import pdb

def test(expdir):
    '''does everything for testing'''
    
    #read the database config file
    database_cfg = configparser.ConfigParser()
    database_cfg.read(os.path.join(expdir, 'database.cfg'))

    #load the model
    with open(os.path.join(expdir, 'model', 'model.pkl'), 'rb') as fid:
        model = pickle.load(fid)

    #read the evaluator config file
    evaluator_cfg = configparser.ConfigParser()
    evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

    #create the evaluator
    evaltype = evaluator_cfg.get('evaluator', 'evaluator')
    evaluator = evaluator_factory.factory(evaltype)(
        conf=evaluator_cfg,
        dataconf=database_cfg,
        model=model)
    
    #create the reconstructor
    reconstruct_type = evaluator_cfg.get('reconstructor', 'reconstruct_type')
    reconstructor = reconstructor_factory.factory(reconstruct_type)(
        conf=evaluator_cfg,
        dataconf=database_cfg,
        expdir=expdir)
	     
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #create the graph
    graph = tf.Graph()

    with graph.as_default():
        #compute the loss
        batch_loss, numbatches, batch_outputs, batch_seq_length = evaluator.evaluate()

        #create a hook that will load the model
        load_hook = LoadAtBegin(
            os.path.join(expdir, 'model', 'network.ckpt'),
            model)

        #create a hook for summary writing
        summary_hook = SummaryHook(os.path.join(expdir, 'logdir'))

        #start the session
        with tf.train.SingularMonitoredSession(
            hooks=[load_hook, summary_hook]) as sess:

            loss = 0.0

            for batch_ind in range(0,numbatches):
		print 'evaluating batch number %d' %batch_ind

		batch_loss_eval, batch_outputs_eval, batch_seq_length_eval = sess.run(
		      fetches=[batch_loss, batch_outputs, batch_seq_length])

                loss += batch_loss_eval

                reconstructor(batch_outputs_eval['outputs'],
			      batch_seq_length_eval['features'])              
                
            loss = loss#/numbatches

    print 'loss = %0.6g' % loss
    
    #write the loss to disk
    with open(os.path.join(expdir, 'loss'), 'w') as fid:
        fid.write(str(loss))
        
    #from here on there is no need for a GPU anymore ==> score script to be run separately on
    #different machine? reconstructor.rec_dir has to be known though. can be put in evaluator_cfg
    
    score_type = evaluator_cfg.get('scorer', 'score_type')
    
    for _ in range(5):
	# Sometime it fails and not sure why. Just retry then. max 5 times
	try:
	    #create the scorer
	    scorer = scorer_factory.factory(score_type)(
		conf=evaluator_cfg,
		dataconf=database_cfg,
		rec_dir=reconstructor.rec_dir,
		numbatches=numbatches)
    
	    #run the scorer
	    scorer()
	except:
	  continue
	break
    
    with open(os.path.join(expdir, 'results_complete.json'), 'w') as fid:
        json.dump(scorer.results,fid)
    
    result_summary = scorer.summarize()
    with open(os.path.join(expdir, 'results_summary.json'), 'w') as fid:
        json.dump(result_summary,fid)


if __name__ == '__main__':

    tf.app.flags.DEFINE_string('expdir', 'expdir',
                               'the exeriments directory that was used for'
                               ' training'
                              )
    FLAGS = tf.app.flags.FLAGS

    test(FLAGS.expdir)
