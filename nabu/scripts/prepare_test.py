'''@file test.py
this file will run the test script'''

import sys
import os
sys.path.append(os.getcwd())
import shutil
import subprocess
from six.moves import configparser
import tensorflow as tf
from test import test
import pdb


tf.app.flags.DEFINE_string('expdir', None,
                           'the exeriments directory that was used for training'
                          )
tf.app.flags.DEFINE_string('recipe', None,
                           'The directory containing the recipe'
                          )
tf.app.flags.DEFINE_string('computing', 'standard',
                           'the distributed computing system one of standart or'
                           ' condor'
                          )

FLAGS = tf.app.flags.FLAGS

def main(_):
    '''main'''

    if FLAGS.expdir is None:
        raise Exception('no expdir specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(FLAGS.expdir):
        raise Exception('cannot find expdir %s' % FLAGS.expdir)

    if FLAGS.recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(FLAGS.recipe):
        raise Exception('cannot find recipe %s' % FLAGS.recipe)
          
    evaluator_cfg_file = os.path.join(FLAGS.recipe, 'test_evaluator.cfg')
    database_cfg_file = os.path.join(FLAGS.recipe, 'database.conf')
    reconstructor_cfg_file = os.path.join(FLAGS.recipe, 'reconstructor.cfg')
    scorer_cfg_file = os.path.join(FLAGS.recipe, 'scorer.cfg')
    postprocessor_cfg_file = os.path.join(FLAGS.recipe, 'postprocessor.cfg')
    model_cfg_file = os.path.join(FLAGS.recipe, 'model.cfg')
    
    #Assuming only one (the last one) training stage needs testing 
    parsed_evaluator_cfg = configparser.ConfigParser()
    parsed_evaluator_cfg.read(evaluator_cfg_file)
    training_stage = parsed_evaluator_cfg.get('evaluator','segment_length')
			  
    ##create the testing dir
    #if os.path.isdir(os.path.join(FLAGS.expdir, 'test')):
        #shutil.rmtree(os.path.join(FLAGS.expdir, 'test'))
    #os.makedirs(os.path.join(FLAGS.expdir, 'test'))
    if not os.path.isdir(os.path.join(FLAGS.expdir, 'test')):
        os.makedirs(os.path.join(FLAGS.expdir, 'test'))
    
    #copy the config files
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(database_cfg_file)	
    segment_parsed_database_cfg = parsed_database_cfg
    
    for section in segment_parsed_database_cfg.sections():
	if 'store_dir' in dict(segment_parsed_database_cfg.items(section)).keys():
	    segment_parsed_database_cfg.set(section,'store_dir',
		    os.path.join(segment_parsed_database_cfg.get(section,'store_dir'),
		    training_stage) )
    with open(os.path.join(FLAGS.expdir, 'test', 'database.cfg'), 'w') as fid:
	segment_parsed_database_cfg.write(fid)
    
    #shutil.copyfile(database_cfg_file,
                    #os.path.join(FLAGS.expdir, 'test', 'database.cfg'))
    shutil.copyfile(evaluator_cfg_file,
                    os.path.join(FLAGS.expdir, 'test', 'evaluator.cfg'))
    shutil.copyfile(reconstructor_cfg_file,
                    os.path.join(FLAGS.expdir, 'test', 'reconstructor.cfg'))
    shutil.copyfile(scorer_cfg_file,
                    os.path.join(FLAGS.expdir, 'test', 'scorer.cfg'))

    try:
	shutil.copyfile(postprocessor_cfg_file,
                    os.path.join(FLAGS.expdir, 'test', 'postprocessor.cfg'))
    except:
	pass
    shutil.copyfile(model_cfg_file,
                    os.path.join(FLAGS.expdir, 'test', 'model.cfg'))

    #create a link to the model that will be used for testing. Assuming
    #it is stored in the 'full' directory of expdir
    if not os.path.isdir(os.path.join(FLAGS.expdir, 'test', 'model')):
	os.symlink(os.path.join(FLAGS.expdir, training_stage, 'model'),
		  os.path.join(FLAGS.expdir, 'test', 'model'))

    if FLAGS.computing == 'condor':

        computing_cfg_file = 'config/computing/condor/non_distributed.cfg'
        parsed_computing_cfg = configparser.ConfigParser()
        parsed_computing_cfg.read(computing_cfg_file)
        computing_cfg = dict(parsed_computing_cfg.items('computing'))

        if not os.path.isdir(os.path.join(FLAGS.expdir, 'test', 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'test', 'outputs'))

        subprocess.call(['condor_submit',
                         'expdir=%s' % os.path.join(FLAGS.expdir, 'test'),
                         'script=nabu/scripts/test.py',
                         'nabu/computing/condor/non_distributed_cpu.job'])


    elif FLAGS.computing == 'standard':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        test(expdir=os.path.join(FLAGS.expdir, 'test'))

    else:
        raise Exception('Unknown computing type %s' % FLAGS.computing)

if __name__ == '__main__':
    tf.app.run()
