'''@file hooks.py
contains session hooks'''

import tensorflow as tf
import warnings
import pdb
from nabu.neuralnetworks.models import run_multi_model

class LoadAtBegin(tf.train.SessionRunHook):
    '''a training hook for loading models at beginning of training'''

    def __init__(self, filename, models):
        '''hook constructor

        Args:
            filename: where the models will be loaded from
            models: the models that will be loaded'''

        self.filename = filename
        self.models = models

    def begin(self):
        '''this will be run at session creation'''
	
        #pylint: disable=W0201
        self._saver = tf.train.Saver(run_multi_model.get_variables(self.models), sharded=True,
				     name='LoaderAtBegin')

    def after_create_session(self, session, _):
        '''this will be run after session creation'''

        self._saver.restore(session, self.filename)

class SummaryHook(tf.train.SessionRunHook):
    '''a training hook for logging the summaries'''

    def __init__(self, logdir):
        '''hook constructor

        Args:
            logdir: logdir where the summaries will be logged'''

        self.logdir = logdir

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._summary = tf.summary.merge_all()


    def after_create_session(self, session, _):
        '''this will be run after session creation'''

        #pylint: disable=W0201
        self._writer = tf.summary.FileWriter(self.logdir, session.graph)

    def before_run(self, _):
        '''this will be executed before a session run call'''

        return tf.train.SessionRunArgs(fetches={'summary':self._summary})

    def after_run(self, _, run_values):
        '''this will be executed after a run call'''

        self._writer.add_summary(run_values.results['summary'])

class SaveAtEnd(tf.train.SessionRunHook):
    '''a training hook for saving the final models'''

    def __init__(self, filename, models):
        '''hook constructor

        Args:
            filename: where the model will be saved
            models: the models that will be saved'''

        self.filename = filename
        self.models = models

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(run_multi_model.get_variables(self.models), sharded=True,
				     name='SaverAtEnd')

    def end(self, session):
        '''this will be run at session closing'''

        self._saver.save(session, self.filename)

class ValidationSaveHook(tf.train.SessionRunHook):
    '''a training hook for saving and loading the validated models'''
    def __init__(self, filename, models):
        '''hook constructor

        Args:
            filename: where the model will be saved
            models: the models that will be saved'''

        self.filename = filename
        self.models = models

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(sharded=True,
				     name='SaverValidation')

    def after_create_session(self, session, _):
        '''this will be run after session creation'''

        #pylint: disable=W0201
        self._sess = session
        
    def save(self):
        '''save the current parameters'''

        self._saver.save(self._sess, self.filename)

    def restore(self):
        '''restore the previously validate parameters'''

        self._saver.restore(self._sess, self.filename)


class StopHook(tf.train.SessionRunHook):
    '''a hook that makes sure all replicas terminate when session ends'''

    def __init__(self, done_op):
        '''hook constructor'''

        self.done_op = done_op

    def end(self, session):
        '''this will be run at session closing'''

        self.done_op.run(session=session)
