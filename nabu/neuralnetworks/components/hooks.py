"""@file hooks.py
contains session hooks"""

import tensorflow as tf
import re
import os
from nabu.neuralnetworks.models import run_multi_model
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training import basic_session_run_hooks

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export

class LoadAtBegin(tf.train.SessionRunHook):
    """a training hook for loading models at beginning of training"""

    def __init__(self, filename, models):
        """hook constructor

        Args:
            filename: where the models will be loaded from
            models: the models that will be loaded"""

        self.filename = filename
        self.models = models

        reader = pywrap_tensorflow.NewCheckpointReader(self.filename)
        vars_and_shapes_in_file = reader.get_variable_to_shape_map()
        self.var_names_in_file = vars_and_shapes_in_file.keys()
        self.var_shapes_in_file = vars_and_shapes_in_file.values()

    def begin(self):
        """this will be run at session creation"""

        # pylint: disable=W0201
        # first find all tensor names in the checkpoint file
        all_requested_variables = run_multi_model.get_variables(self.models)
        requested_names = [var.name for var in all_requested_variables]
        requested_names = [re.sub(':0$', '', name) for name in requested_names]  # remove the trailing :0.
        requested_shapes = [map(int, var.shape) for var in all_requested_variables]

        tmp = [False] * len(all_requested_variables)
        for ind, (name, shape) in enumerate(zip(requested_names, requested_shapes)):
            if name in self.var_names_in_file:
                ind_match = [ind2 for ind2, name_in_file in enumerate(self.var_names_in_file) if name == name_in_file]
                ind_match = ind_match[0]

                if shape == self.var_shapes_in_file[ind_match]:
                    tmp[ind] = True

        requested_variables_in_file = [all_requested_variables[ind] for ind, val in enumerate(tmp) if val]

        if len(requested_variables_in_file) > 0:
            print 'The folowing variables will be loaded from the init file:'
            for var in requested_variables_in_file:
                print var.name


        self._saver = tf.train.Saver(requested_variables_in_file, sharded=True, name='LoaderAtBegin')

    def after_create_session(self, session, _):
        """this will be run after session creation"""

        self._saver.restore(session, self.filename)


# class SummaryHook(tf.train.SessionRunHook):
#     """a training hook for logging the summaries"""
#
#     def __init__(self, logdir):
#         """hook constructor
#
#         Args:
#             logdir: logdir where the summaries will be logged"""
#
#         self.logdir = logdir
#
#     def begin(self):
#         """this will be run at session creation"""
#
#         # pylint: disable=W0201
#         self._summary = tf.summary.merge_all()
#
#     def after_create_session(self, session, _):
#         """this will be run after session creation"""
#
#         # pylint: disable=W0201
#         self._writer = tf.summary.FileWriter(self.logdir, session.graph)
#
#     def before_run(self, _):
#         """this will be executed before a session run call"""
#
#         return tf.train.SessionRunArgs(fetches={'summary':self._summary})
#
#     def after_run(self, _, run_values):
#         """this will be executed after a run call"""
#
#         self._writer.add_summary(run_values.results['summary'])


class SaveAtEnd(tf.train.SessionRunHook):
    """a training hook for saving the final models"""

    def __init__(self, filename, models, should_save_final_model=True):
        """hook constructor

        Args:
            filename: where the model will be saved
            models: the models that will be saved
            should_save_final_model: whether the model should be saved"""

        self.filename = filename
        self.models = models
        self.should_save_final_model = should_save_final_model

    def begin(self):
        """this will be run at session creation"""

        # pylint: disable=W0201
        self._saver = tf.train.Saver(run_multi_model.get_variables(self.models), sharded=True,
                                     name='SaverAtEnd')

    def end(self, session):
        """this will be run at session closing"""

        if self.should_save_final_model is True or session.run(self.should_save_final_model):
            self._saver.save(session, self.filename)


class ValidationSaveHook(tf.train.SessionRunHook):
    """a training hook for saving and loading the validated models"""
    def __init__(self, filename, models):
        """hook constructor

        Args:
            filename: where the model will be saved
            models: the models that will be saved"""

        self.filename = filename
        self.models = models

    def begin(self):
        """this will be run at session creation"""

        # pylint: disable=W0201
        self._saver = tf.train.Saver(sharded=True,
                                     name='SaverValidation')

    def after_create_session(self, session, _):
        """this will be run after session creation"""

        # pylint: disable=W0201
        self._sess = session
        
    def save(self):
        """save the current parameters"""

        self._saver.save(self._sess, self.filename)

    def restore(self):
        """restore the previously validate parameters"""

        self._saver.restore(self._sess, self.filename)


class StopHook(tf.train.SessionRunHook):
    """a hook that makes sure all replicas terminate when session ends"""

    def __init__(self, done_op):
        """hook constructor"""

        self.done_op = done_op

    def end(self, session):
        """this will be run at session closing"""

        self.done_op.run(session=session)


class CheckpointSaverHook(basic_session_run_hooks.CheckpointSaverHook):
  """Saves checkpoints every N steps or seconds."""

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use CheckpointSaverHook.")
    for l in self._listeners:
      l.begin()

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._timer.last_triggered_step() is None:
      # We do write graph and saver_def at the first call of before_run.
      # We cannot do this in begin, since we let other hooks to change graph and
      # add variables in begin. Graph is finalized after all begin calls.
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir,
          "graph.pbtxt")
      saver_def = self._get_saver().saver_def if self._get_saver() else None
      graph = ops.get_default_graph()
      meta_graph_def = meta_graph.create_meta_graph_def(
          graph_def=graph.as_graph_def(add_shapes=True),
          saver_def=saver_def)

    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results
    if self._timer.should_trigger_for_step(
        stale_global_step + self._steps_per_run):
      # get the real value after train op.
      global_step = run_context.session.run(self._global_step_tensor)
      if self._timer.should_trigger_for_step(global_step):
        self._timer.update_last_triggered_step(global_step)
        self._save(run_context.session, global_step)

  def end(self, session):
    last_step = session.run(self._global_step_tensor)
    if last_step != self._timer.last_triggered_step():
      self._save(session, last_step)
    for l in self._listeners:
      l.end(session, last_step)

  def _save(self, session, step):
    """Saves the latest checkpoint."""
    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

    for l in self._listeners:
      l.before_save(session, step)

    self._get_saver().save(session, self._save_path, global_step=step)

    for l in self._listeners:
      l.after_save(session, step)

  def _get_saver(self):
    if self._saver is not None:
      return self._saver
    elif self._scaffold is not None:
      return self._scaffold.saver

    # Get saver from the SAVERS collection if present.
    collection_key = ops.GraphKeys.SAVERS
    savers = ops.get_collection(collection_key)
    if not savers:
      raise RuntimeError(
          "No items in collection {}. Please add a saver to the collection "
          "or provide a saver or scaffold.".format(collection_key))
    elif len(savers) > 1:
      raise RuntimeError(
          "More than one item in collection {}. "
          "Please indicate which one to use by passing it to the constructor.".
          format(collection_key))

    self._saver = savers[0]
    return savers[0]