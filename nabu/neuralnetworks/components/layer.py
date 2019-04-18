"""@file layer.py
Neural network layers """

import string

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from nabu.neuralnetworks.components import ops, rnn_cell, rnn, rnn_cell_impl
from ops import capsule_initializer
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import pdb

_alphabet_str=string.ascii_lowercase


class Capsule(tf.layers.Layer):
    """a capsule layer"""

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_initializer=None,
            logits_initializer=None,
            logits_prior=False,
            routing_iters=3,
            activation_fn=None,
            probability_fn=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            **kwargs):

        """Capsule layer constructor
        args:
            num_capsules: number of output capsules
            capsule_dim: output capsule dimsension
            kernel_initializer: an initializer for the prediction kernel
            logits_initializer: the initializer for the initial logits
            routing_iters: the number of routing iterations (default: 5)
            activation_fn: a callable activation function (default: squash)
            probability_fn: a callable that takes in logits and returns weights
                (default: tf.nn.softmax)
            activity_regularizer: Regularizer instance for the output (callable)
            trainable: wether layer is trainable
            name: the name of the layer
        """

        super(Capsule, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_initializer = kernel_initializer or capsule_initializer()
        self.logits_initializer = logits_initializer or tf.zeros_initializer()
        self.logits_prior = logits_prior
        self.routing_iters = routing_iters
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        """creates the variables of this layer
        args:
            input_shape: the shape of the input
        """

        # pylint: disable=W0201

        # input dimensions
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        self.kernel = self.add_variable(
            name='kernel',
            dtype=self.dtype,
            shape=[num_capsules_in, capsule_dim_in,
                   self.num_capsules, self.capsule_dim],
            initializer=self.kernel_initializer)

        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=self.logits_prior
        )

        super(Capsule, self).build(input_shape)

    # pylint: disable=W0221
    def call(self, inputs):
        """
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        """

        # compute the predictions
        predictions, logits = self.predict(inputs)

        # cluster the predictions
        outputs = self.cluster(predictions, logits)

        return outputs

    def predict(self, inputs):
        """
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        """

        with tf.name_scope('predict'):

            # number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-2

            # put the input capsules as the first dimension
            inputs = tf.transpose(inputs, [shared] + range(shared) + [rank-1])

            # compute the predictins
            predictions = tf.map_fn(
                fn=lambda x: tf.tensordot(x[0], x[1], [[shared], [0]]),
                elems=(inputs, self.kernel),
                dtype=self.dtype or tf.float32)

            # transpose back
            predictions = tf.transpose(
                predictions, range(1, shared+1)+[0]+[rank-1, rank])

            logits = self.logits
            for i in range(shared):
                if predictions.shape[shared-i-1].value is None:
                    shape = tf.shape(predictions)[shared-i-1]
                else:
                    shape = predictions.shape[shared-i-1].value
                tile = [shape] + [1]*len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def predict_slow(self, inputs):
        """
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        """

        with tf.name_scope('predict'):

            # number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-2

            if shared > 26-4:
                raise ValueError('Not enough letters in the alphabet to use Einstein notation')
            # input_shape = [shared (typicaly batch_size,time),Nin,Din], kernel_shape = [Nin, Din, Nout, Dout],
            # predictions_shape = [shared,Nin,Nout,Dout]
            shared_shape_str = _alphabet_str[0:shared]
            input_shape_str = shared_shape_str+'wx'
            kernel_shape_str = 'wxyz'
            output_shape_str = shared_shape_str+'wyz'
            ein_not = '%s,%s->%s' % (input_shape_str, kernel_shape_str, output_shape_str)

            predictions = tf.einsum(ein_not, inputs, self.kernel)

            logits = self.logits
            for i in range(shared):
                if predictions.shape[shared-i-1].value is None:
                    shape = tf.shape(predictions)[shared-i-1]
                else:
                    shape = predictions.shape[shared-i-1].value
                tile = [shape] + [1]*len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def cluster(self, predictions, logits):
        """cluster the predictions into output capsules
        args:
            predictions: the predicted output capsules
            logits: the initial routing logits
        returns:
            the output capsules
        """

        with tf.name_scope('cluster'):

            # define m-step
            def m_step(l):
                """m step"""
                with tf.name_scope('m_step'):
                    # compute the capsule contents
                    w = self.probability_fn(l)
                    caps = tf.reduce_sum(
                        tf.expand_dims(w, -1)*predictions, -3)

                return caps, w

            # define body of the while loop
            def body(l):
                """body"""

                caps, _ = m_step(l)
                caps = self.activation_fn(caps)

                # compare the capsule contents with the predictions
                similarity = tf.reduce_sum(
                    predictions*tf.expand_dims(caps, -3), -1)

                return l + similarity

            # get the final logits with the while loop
            lo = tf.while_loop(
                lambda l: True,
                body, [logits],
                maximum_iterations=self.routing_iters)

            # get the final output capsules
            capsules, _ = m_step(lo)
            capsules = self.activation_fn(capsules)

        return capsules

    def compute_output_shape(self, input_shape):
        """compute the output shape"""

        if input_shape[-2].value is None:
            raise ValueError(
                'The number of capsules must be defined, but saw: %s'
                % input_shape)
        if input_shape[-1].value is None:
            raise ValueError(
                'The capsule dimension must be defined, but saw: %s'
                % input_shape)

        return input_shape[:-2].concatenate(
            [self.num_capsules, self.capsule_dim])


class BRCapsuleLayer(object):
    """a Bidirectional recurrent capsule layer"""

    def __init__(self, num_capsules, capsule_dim, routing_iters=3, activation=None, input_probability_fn=None,
                 recurrent_probability_fn=None, rec_only_vote=False, logits_prior=False, accumulate_input_logits=True,
                 accumulate_state_logits=True):
        """
        BRCapsuleLayer constructor

        Args:
            TODO
        """

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters
        self._activation = activation
        self.input_probability_fn = input_probability_fn
        self.recurrent_probability_fn = recurrent_probability_fn
        self.rec_only_vote = rec_only_vote
        self.logits_prior = logits_prior
        self.accumulate_input_logits = accumulate_input_logits
        self.accumulate_state_logits = accumulate_state_logits

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the rnn cell that will be used for the forward and backward
            # pass
            
            if self.rec_only_vote:
                rnn_cell_fw = rnn_cell.RecCapsuleCellRecOnlyVote(
                    num_capsules=self.num_capsules,
                    capsule_dim=self.capsule_dim,
                    routing_iters=self.routing_iters,
                    activation=self._activation,
                    input_probability_fn=self.input_probability_fn,
                    recurrent_probability_fn=self.recurrent_probability_fn,
                    logits_prior=self.logits_prior,
                    accumulate_input_logits=self.accumulate_input_logits,
                    accumulate_state_logits=self.accumulate_state_logits,
                    reuse=tf.get_variable_scope().reuse)

                rnn_cell_bw = rnn_cell.RecCapsuleCellRecOnlyVote(
                    num_capsules=self.num_capsules,
                    capsule_dim=self.capsule_dim,
                    routing_iters=self.routing_iters,
                    activation=self._activation,
                    input_probability_fn=self.input_probability_fn,
                    recurrent_probability_fn=self.recurrent_probability_fn,
                    logits_prior=self.logits_prior,
                    accumulate_input_logits=self.accumulate_input_logits,
                    accumulate_state_logits=self.accumulate_state_logits,
                    reuse=tf.get_variable_scope().reuse)
            else:
                rnn_cell_fw = rnn_cell.RecCapsuleCell(
                    num_capsules=self.num_capsules,
                    capsule_dim=self.capsule_dim,
                    routing_iters=self.routing_iters,
                    activation=self._activation,
                    input_probability_fn=self.input_probability_fn,
                    recurrent_probability_fn=self.recurrent_probability_fn,
                    logits_prior=self.logits_prior,
                    reuse=tf.get_variable_scope().reuse)

                rnn_cell_bw = rnn_cell.RecCapsuleCell(
                    num_capsules=self.num_capsules,
                    capsule_dim=self.capsule_dim,
                    routing_iters=self.routing_iters,
                    activation=self._activation,
                    input_probability_fn=self.input_probability_fn,
                    recurrent_probability_fn=self.recurrent_probability_fn,
                    logits_prior=self.logits_prior,
                    reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                rnn_cell_fw, rnn_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class BLSTMCapsuleLayer(object):
    """a Bidirectional lstm capsule layer"""

    def __init__(self, num_capsules, capsule_dim, routing_iters=3, activation=None, input_probability_fn=None,
                 recurrent_probability_fn=None,  logits_prior=False, accumulate_input_logits=True,
                 accumulate_state_logits=True, gates_fc = False, use_output_matrix=False):
        """
        BRCapsuleLayer constructor

        Args:
            TODO
        """

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters
        self._activation = activation
        self.input_probability_fn = input_probability_fn
        self.recurrent_probability_fn = recurrent_probability_fn
        self.logits_prior = logits_prior
        self.accumulate_input_logits = accumulate_input_logits
        self.accumulate_state_logits = accumulate_state_logits
        self.gates_fc = gates_fc
        self.use_output_matrix = use_output_matrix

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the rnn cell that will be used for the forward and backward
            # pass

            if self.use_output_matrix:
                lstm_cell_type = rnn_cell.LSTMCapsuleCellRecOnlyVoteOutputMatrix
            else:
                lstm_cell_type = rnn_cell.LSTMCapsuleCellRecOnlyVote

            lstm_cell_fw = lstm_cell_type(
                num_capsules=self.num_capsules,
                capsule_dim=self.capsule_dim,
                routing_iters=self.routing_iters,
                activation=self._activation,
                input_probability_fn=self.input_probability_fn,
                recurrent_probability_fn=self.recurrent_probability_fn,
                logits_prior=self.logits_prior,
                accumulate_input_logits=self.accumulate_input_logits,
                accumulate_state_logits=self.accumulate_state_logits,
                gates_fc=self.gates_fc,
                reuse=tf.get_variable_scope().reuse)

            lstm_cell_bw = lstm_cell_type(
                num_capsules=self.num_capsules,
                capsule_dim=self.capsule_dim,
                routing_iters=self.routing_iters,
                activation=self._activation,
                input_probability_fn=self.input_probability_fn,
                recurrent_probability_fn=self.recurrent_probability_fn,
                logits_prior=self.logits_prior,
                accumulate_input_logits=self.accumulate_input_logits,
                accumulate_state_logits=self.accumulate_state_logits,
                gates_fc=self.gates_fc,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class BRNNLayer(object):
    """a BRNN layer"""

    def __init__(self, num_units, activation_fn=tf.nn.tanh, linear_out_flag=False):
        """
        BRNNLayer constructor

        Args:
            num_units: The number of units in the one directon
            activation_fn: activation function
            linear_out_flag: if set to True, activation function will only be applied
            to the recurrent output.
        """

        self.num_units = num_units
        self.activation_fn = activation_fn
        self.linear_out_flag = linear_out_flag

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the rnn cell that will be used for the forward and backward
            # pass
            if self.linear_out_flag:
                rnn_cell_type = rnn_cell.RNNCellLinearOut
            else:
                rnn_cell_type = tf.contrib.rnn.BasicRNNCell

            rnn_cell_fw = rnn_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
            rnn_cell_bw = rnn_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                rnn_cell_fw, rnn_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class LSTMLayer(object):
    """a LSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
        """
        LSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            activation_fn: activation function
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs, _ = dynamic_rnn(
                lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            return outputs


class BLSTMLayer(object):
    """a BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh,
                 separate_directions=False, linear_out_flag=False, fast_version=False):
        """
        BLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            separate_directions: wether the forward and backward directions should
            be separated for deep networks.
            fast_version: deprecated
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn
        self.separate_directions = separate_directions
        self.linear_out_flag = linear_out_flag
        self.fast_version = fast_version

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass

            if self.linear_out_flag:
                lstm_cell_type = rnn_cell.LayerNormBasicLSTMCellLineairOut
            else:
                lstm_cell_type = tf.contrib.rnn.LayerNormBasicLSTMCell

            lstm_cell_fw = lstm_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw =lstm_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            if not self.separate_directions:
                outputs_tupple, _ = bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                                                              sequence_length=sequence_length)

                outputs = tf.concat(outputs_tupple, 2)
            else:
                outputs, _ = rnn.bidirectional_dynamic_rnn_2inputs(
                    lstm_cell_fw, lstm_cell_bw, inputs[0], inputs[1], dtype=tf.float32,
                    sequence_length=sequence_length)

            return outputs


class LeakyLSTMLayer(object):
    """a leaky LSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell = rnn_cell.LayerNormBasicLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs, _ = dynamic_rnn(
                lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            return outputs


class LeakyBLSTMLayer(object):
    """a leaky BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell_fw = rnn_cell.LayerNormBasicLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormBasicLeakLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  


class LeakyBLSTMIZNotRecLayer(object):
    """a leaky BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMIZNotRecLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell_fw = rnn_cell.LayerNormIZNotRecLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormIZNotRecLeakLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  


class LeakyBLSTMNotRecLayer(object):
    """a leaky BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMNotRecLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell_fw = rnn_cell.LayerNormNotRecLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormNotRecLeakLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  


class LeakychBLSTMLayer(object):
    """a leaky ch BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell_fw = rnn_cell.LayerNormBasicLeakchLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormBasicLeakchLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class ResetLSTMLayer(object):
	"""a ResetLSTM layer"""

	def __init__(
			self, num_units, t_reset=1, next_t_reset=None, layer_norm=False, recurrent_dropout=1.0,
			activation_fn=tf.nn.tanh):
		"""
		ResetLSTM constructor

		Args:
			num_units: The number of units in the one directon
			layer_norm: whether layer normalization should be applied
			recurrent_dropout: the recurrent dropout keep probability
		"""

		self.num_units = num_units
		self.t_reset = t_reset
		if next_t_reset:
			if next_t_reset < t_reset:
				raise ValueError('T_reset in next layer must be equal to or bigger than T_reset in current layer')
			self.next_t_reset = next_t_reset
		else:
			self.next_t_reset = t_reset
		self.layer_norm = layer_norm
		self.recurrent_dropout = recurrent_dropout
		self.activation_fn = activation_fn

	def __call__(self, inputs, sequence_length, scope=None):
		"""
		Create the variables and do the forward computation

		Args:
			inputs: the input to the layer as a
				[batch_size, max_length, dim] tensor
			sequence_length: the length of the input sequences as a
				[batch_size] tensor
			scope: The variable scope sets the namespace under which
				the variables created during this call will be stored.

		Returns:
			the output of the layer
		"""
		batch_size = inputs.get_shape()[0]
		max_length = tf.shape(inputs)[1]

		with tf.variable_scope(scope or type(self).__name__):

			# create the lstm cell that will be used for the forward
			lstm_cell = rnn_cell.LayerNormResetLSTMCell(
				num_units=self.num_units,
				t_reset=self.t_reset,
				activation=self.activation_fn,
				layer_norm=self.layer_norm,
				dropout_keep_prob=self.recurrent_dropout,
				reuse=tf.get_variable_scope().reuse)

			# do the forward computation
			outputs_tupple, _ = rnn.dynamic_rnn_time_input(
				lstm_cell, inputs, dtype=tf.float32,
				sequence_length=sequence_length)

			if self.next_t_reset == self.t_reset:
				return outputs_tupple

			actual_outputs = outputs_tupple[0]
			replicas = outputs_tupple[1]

			# the output replicas need to be permuted correctly such that the next layer receives
			# the replicas in the correct order

			# numbers_to_maxT: [1, Tmax,1]
			numbers_to_maxT = tf.range(0, max_length)
			numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, -1), 0)

			# numbers_to_k: [1, 1,k]
			numbers_to_k = tf.expand_dims(tf.expand_dims(range(0, self.next_t_reset), 0), 0)

			mod1 = tf.mod(numbers_to_maxT - 1 - numbers_to_k, self.next_t_reset)
			mod2 = tf.mod(numbers_to_maxT - mod1 - 1, self.t_reset)
			mod3 = tf.tile(tf.mod(numbers_to_maxT, self.t_reset), [1, 1, self.next_t_reset])

			indices_for_next_layer = tf.where(
				mod1 < self.t_reset,
				x=mod2,
				y=mod3,
			)
			indices_for_next_layer = tf.tile(indices_for_next_layer, [batch_size, 1, 1])

			# ra1: [B,Tmax,k]
			ra1 = tf.range(batch_size)
			ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
			ra1 = tf.tile(ra1, [1, max_length, self.next_t_reset])
			ra2 = tf.range(max_length)
			ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
			ra2 = tf.tile(ra2, [batch_size, 1, self.next_t_reset])
			full_indices_for_next_layer = tf.stack([ra1, ra2, indices_for_next_layer], axis=-1)
			output_for_next_layer = tf.gather_nd(replicas, full_indices_for_next_layer)

			outputs = (actual_outputs, output_for_next_layer)

		return outputs


class BResetLSTMLayer(object):
	"""a BResetLSTM layer"""

	def __init__(
			self, num_units, t_reset=1, next_t_reset=None, group_size=1, symmetric_context=False, layer_norm=False,
			recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
		"""
		BResetLSTM constructor

		Args:
			num_units: The number of units in the one directon
			group_size: units in the same group share a state replicate
			symmetric_context: if True, input to next layer should have same amount of context
			in both directions. If False, reversed input to next layers has full (t_reset) context.
			layer_norm: whether layer normalization should be applied
			recurrent_dropout: the recurrent dropout keep probability
		"""

		self.num_units = num_units
		self.t_reset = t_reset
		if next_t_reset:
			if next_t_reset < t_reset:
				raise ValueError('T_reset in next layer must be equal to or bigger than T_reset in current layer')
			self.next_t_reset = next_t_reset
		else:
			self.next_t_reset = t_reset
		self.group_size = group_size
		if self.group_size > 1 and self.next_t_reset != self.t_reset:
			raise NotImplementedError('Grouping is not yet implemented for different t_resets')
		self.num_replicates = float(self.t_reset)/float(self.group_size)
		if int(self.num_replicates) != self.num_replicates:
			raise ValueError('t_reset should be a multiple of group_size')
		self.symmetric_context = symmetric_context
		self.num_replicates = int(self.num_replicates)
		self.layer_norm = layer_norm
		self.recurrent_dropout = recurrent_dropout
		self.activation_fn = activation_fn

	def __call__(self, inputs_for_forward, inputs_for_backward, sequence_length, scope=None):
		"""
		Create the variables and do the forward computation

		Args:
			inputs: the input to the layer as a
				[batch_size, max_length, dim] tensor
			sequence_length: the length of the input sequences as a
				[batch_size] tensor
			scope: The variable scope sets the namespace under which
				the variables created during this call will be stored.

		Returns:
			the output of the layer
		"""

		if inputs_for_backward is None:
			inputs_for_backward = inputs_for_forward

		batch_size = inputs_for_forward.get_shape()[0]
		max_length = tf.shape(inputs_for_forward)[1]

		with tf.variable_scope(scope or type(self).__name__):
			# create the lstm cell that will be used for the forward and backward
			# pass
			if self.group_size == 1:
				lstm_cell_fw = rnn_cell.LayerNormResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				lstm_cell_bw = rnn_cell.LayerNormResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				tile_shape = [1, 1, self.t_reset, 1]
			else:
				lstm_cell_fw = rnn_cell.LayerNormGroupResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					group_size=self.group_size,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				lstm_cell_bw = rnn_cell.LayerNormGroupResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					group_size=self.group_size,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				tile_shape = [1, 1, lstm_cell_fw._num_replicates, 1]

			# do the forward computation
			outputs_tupple, _ = rnn.bidirectional_dynamic_rnn_2inputs_time_input(
				lstm_cell_fw, lstm_cell_bw, inputs_for_forward, inputs_for_backward,
				dtype=tf.float32, sequence_length=sequence_length)

			# outputs are reordered for next layer.
			# TODO:This should be done in model.py and not in layer.py
			actual_outputs_forward = outputs_tupple[0][0]
			actual_outputs_backward = outputs_tupple[1][0]
			actual_outputs = tf.concat((actual_outputs_forward, actual_outputs_backward), -1)

			forward_replicas = outputs_tupple[0][1]
			backward_replicas = outputs_tupple[1][1]

			if not self.symmetric_context:
				forward_for_backward = tf.expand_dims(actual_outputs_forward, -2)
				forward_for_backward = tf.tile(forward_for_backward, tile_shape)

				backward_for_forward = tf.expand_dims(actual_outputs_backward, -2)
				backward_for_forward = tf.tile(backward_for_forward, tile_shape)

				outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
				outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

			elif False and self.t_reset == self.next_t_reset:
				# the output replicas need to be permuted correctly such that the next layer receives
				# the replicas in the correct order

				# T_min_1: [B,1]
				T = tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size)))
				T_min_1 = tf.expand_dims(T - 1, -1)

				# numbers_to_maxT: [1,Tmax,1]
				numbers_to_maxT = tf.range(0, max_length)
				numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, 0), -1)

				# numbers_to_k: [1,k]
				numbers_to_k = tf.expand_dims(range(0, self.num_replicates), 0)

				# backward_indices_for_forward_t_0: [B,1,k]
				backward_indices_for_forward_t_0 = numbers_to_k+T_min_1
				# backward_indices_for_forward_t_0 = tf.mod(backward_indices_for_forward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
				backward_indices_for_forward_t_0 = tf.expand_dims(backward_indices_for_forward_t_0, 1)
				# backward_indices_for_forward_t: [B,Tmax,k]
				backward_indices_for_forward_t = tf.mod(backward_indices_for_forward_t_0 - 2*numbers_to_maxT,
														self.num_replicates)

				forward_indices_for_backward_t_0 = numbers_to_k-T_min_1
				# forward_indices_for_backward_t_0 = tf.mod(forward_indices_for_backward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
				forward_indices_for_backward_t_0 = tf.expand_dims(forward_indices_for_backward_t_0, 1)
				forward_indices_for_backward_t = tf.mod(forward_indices_for_backward_t_0 + 2*numbers_to_maxT,
														self.num_replicates)

				# ra1: [B,Tmax,k]
				ra1 = tf.range(batch_size)
				ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
				ra1 = tf.tile(ra1, [1, max_length, self.num_replicates])
				ra2 = tf.range(max_length)
				ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
				ra2 = tf.tile(ra2, [batch_size, 1, self.num_replicates])
				stacked_backward_indices_for_forward_t = tf.stack([ra1, ra2, backward_indices_for_forward_t], axis=-1)
				backward_for_forward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_forward_t)
				stacked_forward_indices_for_backward_t = tf.stack([ra1, ra2, forward_indices_for_backward_t], axis=-1)
				forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward_t)

				outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
				outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

			else:
				# the output replicas need to be permuted correctly such that the next layer receives
				# the replicas in the correct order

				# T: [B,1, 1]
				T = tf.expand_dims(tf.expand_dims(
					tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size))), -1), -1)

				# numbers_to_maxT: [B,Tmax,k]
				numbers_to_maxT = tf.range(0, max_length)
				numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, 0), -1)
				numbers_to_maxT = tf.tile(numbers_to_maxT, [batch_size, 1, self.next_t_reset])
				reversed_numbers_to_maxT = T - 1 - numbers_to_maxT

				# numbers_to_k: [B,Tmax,k]
				numbers_to_k = tf.expand_dims(tf.expand_dims(range(0, self.next_t_reset), 0), 0)
				numbers_to_k = tf.tile(numbers_to_k, [batch_size, max_length, 1])

				# next taus
				next_tau_forward = tf.mod(numbers_to_maxT-1-numbers_to_k, self.next_t_reset)
				next_tau_backward = tf.mod(reversed_numbers_to_maxT-1-numbers_to_k, self.next_t_reset)

				# max memory instances
				max_memory_forward = tf.mod(numbers_to_maxT, self.t_reset)
				max_memory_backward = tf.mod(reversed_numbers_to_maxT, self.t_reset)

				# forward for forward
				condition_forward = next_tau_forward < self.t_reset
				condition_true = tf.mod(numbers_to_maxT-1-next_tau_forward, self.t_reset)
				forward_indices_for_forward = tf.where(condition_forward, x=condition_true, y=max_memory_forward)

				# backward for forward
				condition_true = tf.mod(reversed_numbers_to_maxT-1-next_tau_forward, self.t_reset)
				backward_indices_for_forward = tf.where(condition_forward, x=condition_true, y=max_memory_backward)

				# backward for backward
				condition_backward = next_tau_backward < self.t_reset
				condition_true = tf.mod(reversed_numbers_to_maxT-1-next_tau_backward, self.t_reset)
				backward_indices_for_backward = tf.where(condition_backward, x=condition_true, y=max_memory_backward)

				# forward for backward
				condition_true = tf.mod(numbers_to_maxT-1-next_tau_backward, self.t_reset)
				forward_indices_for_backward = tf.where(condition_backward, x=condition_true, y=max_memory_forward)

				# ra1: [B,Tmax,k]
				ra1 = tf.range(batch_size)
				ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
				ra1 = tf.tile(ra1, [1, max_length, self.next_t_reset])
				ra2 = tf.range(max_length)
				ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
				ra2 = tf.tile(ra2, [batch_size, 1, self.next_t_reset])
				stacked_forward_indices_for_forward = tf.stack([ra1, ra2, forward_indices_for_forward], axis=-1)
				forward_for_forward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_forward)
				stacked_backward_indices_for_forward = tf.stack([ra1, ra2, backward_indices_for_forward], axis=-1)
				backward_for_forward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_forward)
				stacked_backward_indices_for_backward = tf.stack([ra1, ra2, backward_indices_for_backward], axis=-1)
				backward_for_backward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_backward)
				stacked_forward_indices_for_backward = tf.stack([ra1, ra2, forward_indices_for_backward], axis=-1)
				forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward)

				outputs_for_forward = tf.concat((forward_for_forward, backward_for_forward), -1)
				outputs_for_backward = tf.concat((forward_for_backward, backward_for_backward), -1)

		outputs = (actual_outputs, outputs_for_forward, outputs_for_backward)

		return outputs


class BGRULayer(object):
    """a BGRU layer"""

    def __init__(self, num_units, activation_fn=tf.nn.tanh):
        """
        BGRULayer constructor

        Args:
            num_units: The number of units in the one directon
        """

        self.num_units = num_units
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the gru cell that will be used for the forward and backward
            # pass
            gru_cell_fw = tf.contrib.rnn.GRUCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
            gru_cell_bw = tf.contrib.rnn.GRUCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                gru_cell_fw, gru_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class LeakyBGRULayer(object):
    """a leaky BGRU layer"""

    def __init__(self, num_units, activation_fn=tf.nn.tanh, leak_factor=1.0):
        """
        LeakyBGRULayer constructor

        Args:
            num_units: The number of units in the one directon
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.activation_fn = activation_fn
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the gru cell that will be used for the forward and backward
            # pass
            gru_cell_fw = rnn_cell.LeakGRUCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
            gru_cell_bw = rnn_cell.LeakGRUCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                gru_cell_fw, gru_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class BResetGRULayer(object):
    """a BResetGRU layer"""

    def __init__(self, num_units, t_reset=1, group_size=1, symmetric_context=False, activation_fn=tf.nn.tanh):
        """
        BResetLSTM constructor

        Args:
            num_units: The number of units in the one directon
            group_size: units in the same group share a state replicate
            symmetric_context: if True, input to next layer should have same amount of context
            in both directions. If False, reversed input to next layers has full (t_reset) context.
        """

        self.num_units = num_units
        self.t_reset = t_reset
        self.group_size = group_size
        self.num_replicates = float(self.t_reset)/float(self.group_size)
        if int(self.num_replicates) != self.num_replicates:
            raise ValueError('t_reset should be a multiple of group_size')
        self.symmetric_context = symmetric_context
        self.num_replicates = int(self.num_replicates)
        self.activation_fn = activation_fn

    def __call__(self, inputs_for_forward, inputs_for_backward, sequence_length, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """
        
        if inputs_for_backward is None:
            inputs_for_backward = inputs_for_forward

        batch_size = inputs_for_forward.get_shape()[0]
        max_length = tf.shape(inputs_for_forward)[1]

        with tf.variable_scope(scope or type(self).__name__):
            # create the gru cell that will be used for the forward and backward
            # pass
            if self.group_size == 1:
                gru_cell_fw = rnn_cell_impl.ResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)
                gru_cell_bw = rnn_cell_impl.ResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)

                tile_shape = [1, 1 ,self.t_reset, 1]
            else:
                gru_cell_fw = rnn_cell_impl.GroupResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    group_size = self.group_size,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)
                gru_cell_bw = rnn_cell_impl.GroupResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    group_size = self.group_size,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)

                tile_shape = [1,1,gru_cell_fw._num_replicates,1]

            # do the forward computation
            outputs_tupple, _ = rnn.bidirectional_dynamic_rnn_2inputs_time_input(
                gru_cell_fw, gru_cell_bw, inputs_for_forward, inputs_for_backward, 
                dtype=tf.float32, sequence_length=sequence_length)

            actual_outputs_forward = outputs_tupple[0][0]
            actual_outputs_backward = outputs_tupple[1][0]
            actual_outputs = tf.concat((actual_outputs_forward,actual_outputs_backward), -1)

            forward_replicas = outputs_tupple[0][1]
            backward_replicas = outputs_tupple[1][1]

            if not self.symmetric_context:
                forward_for_backward = tf.expand_dims(actual_outputs_forward,-2)
                forward_for_backward = tf.tile(forward_for_backward, tile_shape)

                backward_for_forward = tf.expand_dims(actual_outputs_backward,-2)
                backward_for_forward = tf.tile(backward_for_forward, tile_shape)

                outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
                outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

            else:
                # the output replicas need to be permutated correclty such that the next layer receives
                # the replicas in the correct order
                T = tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size)))
                T_min_1 = tf.expand_dims(T - 1, -1)

                numbers_to_maxT = tf.range(0, max_length)
                numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT,0),-1)

                numbers_to_k = tf.expand_dims(range(0, self.num_replicates), 0)

                backward_indices_for_forward_t_0 = numbers_to_k+T_min_1
                #backward_indices_for_forward_t_0 = tf.mod(backward_indices_for_forward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
                backward_indices_for_forward_t_0 = tf.expand_dims(backward_indices_for_forward_t_0, 1)
                backward_indices_for_forward_t = tf.mod(backward_indices_for_forward_t_0 - 2*numbers_to_maxT,
                                                        self.num_replicates)

                forward_indices_for_backward_t_0 = numbers_to_k-T_min_1
                #forward_indices_for_backward_t_0 = tf.mod(forward_indices_for_backward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
                forward_indices_for_backward_t_0 = tf.expand_dims(forward_indices_for_backward_t_0, 1)
                forward_indices_for_backward_t = tf.mod(forward_indices_for_backward_t_0 + 2*numbers_to_maxT,
                                                        self.num_replicates)

                ra1 = tf.range(batch_size)
                ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
                ra1 = tf.tile(ra1, [1, max_length, self.num_replicates])
                ra2 = tf.range(max_length)
                ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
                ra2 = tf.tile(ra2, [batch_size, 1, self.num_replicates])
                stacked_backward_indices_for_forward_t = tf.stack([ra1, ra2, backward_indices_for_forward_t], axis=-1)
                backward_for_forward = tf.gather_nd(backward_replicas,  stacked_backward_indices_for_forward_t)
                stacked_forward_indices_for_backward_t = tf.stack([ra1, ra2, forward_indices_for_backward_t], axis=-1)
                forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward_t)

                outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
                outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

            outputs = (actual_outputs, outputs_for_forward, outputs_for_backward)

            return outputs


class Conv2D(object):
    """a Conv2D layer, with max_pool and layer norm options"""

    def __init__(self, num_filters, kernel_size, strides=(1, 1), padding='same', activation_fn=tf.nn.relu,
                 layer_norm=False, max_pool_filter=(1,1), transpose=False):
        """
        BLSTMLayer constructor

        Args:
            num_filters: The number of filters
            kernel_size: kernel filter size
            strides: stride size
            padding: padding algorithm
            activation_fn: hidden unit activation
            layer_norm: whether layer normalization should be applied
            max_pool_filter: pooling filter size
            transpose: if true use tf.layers.conv2d_transpose
        """

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.max_pool_filter = max_pool_filter
        self.transpose = transpose

    def __call__(self, inputs, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim, in_channel] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):
            
            if not self.transpose:
                outputs = tf.layers.conv2d(
                        inputs=inputs,
                        filters=self.num_filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding=self.padding,
                        activation=self.activation_fn)
            else:
                outputs = tf.layers.conv2d_transpose(
                        inputs=inputs,
                        filters=self.num_filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding=self.padding,
                        activation=self.activation_fn)

        if self.layer_norm:
            outputs = tf.layers.batch_normalization(outputs)

        outputs_before_pool = outputs

        if self.max_pool_filter != (1, 1):
            outputs = tf.layers.max_pooling2d(outputs, self.max_pool_filter, strides=self.max_pool_filter,
                                              padding='valid')

        return outputs, outputs_before_pool


def unpool(pool_input, pool_output, unpool_input, pool_kernel_size, pool_stride, padding='VALID'):
    """ An unpooling layer as described in Adaptive Deconvolutional Networks for Mid and High Level Feature Learning
    from Matthew D. Zeiler, Graham W. Taylor and Rob Fergus,
    using the implementation of https://assiaben.github.io/posts/2018-06-tf-unpooling/
    """

    # Assuming pool_kernel_size and pool_stride are given in 'HW' format, converting them to 'NHWC' format
    if len(pool_kernel_size) != 2:
        raise ValueError('Expected kernel size to be in "HW" format.')
    pool_kernel_size = [1] + pool_kernel_size + [1]
    if len(pool_stride) != 2:
        raise ValueError('Expected stride size to be in "HW" format.')
    pool_stride = [1] + pool_stride + [1]

    unpool_op = gen_nn_ops.max_pool_grad(pool_input, pool_output, unpool_input, pool_kernel_size, pool_stride, padding)

    return unpool_op


# @ops.RegisterGradient("MaxPoolGradWithArgmax")
# def _MaxPoolGradGradWithArgmax(op, grad):
#     """Register max pooling gradient for the unpool operation. Copied from
#     https://assiaben.github.io/posts/2018-06-tf-unpooling/
#     """
#     print(len(op.outputs))
#     print(len(op.inputs))
#     print(op.name)
#     return (array_ops.zeros(
#       shape=array_ops.shape(op.inputs[0]),
#       dtype=op.inputs[0].dtype), array_ops.zeros(
#           shape=array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype),
#           gen_nn_ops._max_pool_grad_grad_with_argmax(
#               op.inputs[0],
#               grad,
#               op.inputs[2],
#               op.get_attr("ksize"),
#               op.get_attr("strides"),
#               padding=op.get_attr("padding")))