from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nabu.neuralnetworks.components import hooks as nabu_hooks

import abc
import sys

import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import util
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util.tf_export import tf_export



def MonitoredTrainingSession(master='',  # pylint: disable=invalid-name
                             is_chief=True,
                             checkpoint_dir=None,
                             scaffold=None,
                             hooks=None,
                             chief_only_hooks=None,
                             save_checkpoint_secs=None,
                             save_summaries_steps=None,
                             save_summaries_secs=None,
                             config=None,
                             stop_grace_period_secs=120,
                             log_step_count_steps=100,
                             max_wait_secs=7200,
                             save_checkpoint_steps=None):
  """Creates a `MonitoredSession` for training.

  For a chief, this utility sets proper session initializer/restorer. It also
  creates hooks related to checkpoint and summary saving. For workers, this
  utility sets proper session creator which waits for the chief to
  initialize/restore. Please check `tf.train.MonitoredSession` for more
  information.


  Args:
    master: `String` the TensorFlow master to use.
    is_chief: If `True`, it will take care of initialization and recovery the
      underlying TensorFlow session. If `False`, it will wait on a chief to
      initialize or recover the TensorFlow session.
    checkpoint_dir: A string.  Optional path to a directory where to restore
      variables.
    scaffold: A `Scaffold` used for gathering or building supportive ops. If
      not specified, a default one is created. It's used to finalize the graph.
    hooks: Optional list of `SessionRunHook` objects.
    chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If both `save_checkpoint_steps` and
      `save_checkpoint_secs` are set to `None`, then the default checkpoint
      saver isn't used. If both are provided, then only `save_checkpoint_secs`
      is used. Default 600.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config: an instance of `tf.ConfigProto` proto used to configure the session.
      It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps: The frequency, in number of global steps, that the
      global step/sec is logged.
    max_wait_secs: Maximum time workers should wait for the session to
      become available. This should be kept relatively short to help detect
      incorrect code, but sometimes may need to be increased if the chief takes
      a while to start up.
    save_checkpoint_steps: The frequency, in number of global steps, that a
      checkpoint is saved using a default checkpoint saver. If both
      `save_checkpoint_steps` and `save_checkpoint_secs` are set to `None`, then
      the default checkpoint saver isn't used. If both are provided, then only
      `save_checkpoint_secs` is used. Default not enabled.

  Returns:
    A `MonitoredSession` object.
  """

  save_checkpoint_secs = 1800
  save_checkpoint_steps = None

  all_hooks = []
  if chief_only_hooks:
    all_hooks.extend(chief_only_hooks)
  session_creator = tf.train.ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_dir=checkpoint_dir,
      master=master,
      config=config)

  if checkpoint_dir:
    if log_step_count_steps and log_step_count_steps > 0:
      all_hooks.append(
          basic_session_run_hooks.StepCounterHook(
              output_dir=checkpoint_dir, every_n_steps=log_step_count_steps, summary_writer=tf.summary.FileWriter(checkpoint_dir)))

    if (save_checkpoint_secs and save_checkpoint_secs > 0) or (
        save_checkpoint_steps and save_checkpoint_steps > 0):
      all_hooks.append(nabu_hooks.CheckpointSaverHook(
          checkpoint_dir,
          save_steps=save_checkpoint_steps,
          save_secs=save_checkpoint_secs,
          scaffold=scaffold))

  if hooks:
    all_hooks.extend(hooks)
  return tf.train.MonitoredSession(session_creator=session_creator, hooks=all_hooks,
                          stop_grace_period_secs=stop_grace_period_secs)
