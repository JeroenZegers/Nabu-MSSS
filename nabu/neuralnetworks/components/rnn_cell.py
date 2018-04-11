'''@file rnn_cell.py
contains some customized rnn cells'''

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class LayerNormBasicLeakLSTMCell(rnn_cell_impl.RNNCell):
  """LSTM unit with layer normalization, recurrent dropout and
  memory leakage.

  This class adds layer normalization, recurrent dropout and memory
  leakage to a basic LSTM unit. Layer normalization implementation 
  is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:

    https://arxiv.org/abs/1603.05118

  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):
    """Initializes the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      leak_factor: 
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LayerNormBasicLeakLSTMCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._leak_factor = leak_factor
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = array_ops.concat([inputs, h], 1)
    concat = self._linear(args)

    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
    if self._layer_norm:
      i = self._norm(i, "input")
      j = self._norm(j, "transform")
      f = self._norm(f, "forget")
      o = self._norm(o, "output")

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (c * math_ops.sigmoid(f + self._forget_bias)
             + math_ops.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) * math_ops.sigmoid(o)
    
    new_c = new_c * self._leak_factor

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    return new_h, new_state

class LayerNormIZNotRecLeakLSTMCell(rnn_cell_impl.RNNCell):
  """LSTM unit with layer normalization, recurrent dropout and
  memory leakage. The input gate (i) and the transformed input (z, or j as
  in the code below) are only based on the layers input and not recurrent

  This class adds layer normalization, recurrent dropout and memory
  leakage to a basic LSTM unit. Layer normalization implementation 
  is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:

    https://arxiv.org/abs/1603.05118

  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):
    """Initializes the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      leak_factor: 
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LayerNormIZNotRecLeakLSTMCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._leak_factor = leak_factor
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 2 * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out
  
  def _linear_inonly(self, args):
    out_size = 2 * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel_inonly", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias_inonly", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = array_ops.concat([inputs, h], 1)
    concat = self._linear(args)
    concat_inonly = self._linear_inonly(inputs)

    f, o = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
    i, j = array_ops.split(value=concat_inonly, num_or_size_splits=2, axis=1) 
    if self._layer_norm:
      i = self._norm(i, "input")
      j = self._norm(j, "transform")
      f = self._norm(f, "forget")
      o = self._norm(o, "output")

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (c * math_ops.sigmoid(f + self._forget_bias)
             + math_ops.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) * math_ops.sigmoid(o)
    
    new_c = new_c * self._leak_factor

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    return new_h, new_state


class LayerNormNotRecLeakLSTMCell(rnn_cell_impl.RNNCell):
  """LSTM unit with layer normalization, recurrent dropout and
  memory leakage. All weights are only dependent on cell inputs and not
  on recurrent hidden units

  This class adds layer normalization, recurrent dropout and memory
  leakage to a basic LSTM unit. Layer normalization implementation 
  is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:

    https://arxiv.org/abs/1603.05118

  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):
    """Initializes the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      leak_factor: 
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LayerNormNotRecLeakLSTMCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._leak_factor = leak_factor
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    #args = array_ops.concat([inputs, h], 1)
    concat = self._linear(inputs)

    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1) 
    if self._layer_norm:
      i = self._norm(i, "input")
      j = self._norm(j, "transform")
      f = self._norm(f, "forget")
      o = self._norm(o, "output")

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (c * math_ops.sigmoid(f + self._forget_bias)
             + math_ops.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) * math_ops.sigmoid(o)
    
    new_c = new_c * self._leak_factor

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    return new_h, new_state

class LayerNormBasicLeakchLSTMCell(rnn_cell_impl.RNNCell):
  """LSTM unit with layer normalization, recurrent dropout and
  memory leakage, both the c and h variable.

  This class adds layer normalization, recurrent dropout and memory
  leakage to a basic LSTM unit. Layer normalization implementation 
  is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:

    https://arxiv.org/abs/1603.05118

  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self, num_units, leak_factor=1.0, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):
    """Initializes the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      leak_factor: 
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LayerNormBasicLeakchLSTMCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._leak_factor = leak_factor
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    
    #leak recurrent memory
    c = c * self._leak_factor
    h = h * self._leak_factor
    
    args = array_ops.concat([inputs, h], 1)
    concat = self._linear(args)

    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
    if self._layer_norm:
      i = self._norm(i, "input")
      j = self._norm(j, "transform")
      f = self._norm(f, "forget")
      o = self._norm(o, "output")

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (c * math_ops.sigmoid(f + self._forget_bias)
             + math_ops.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) * math_ops.sigmoid(o)
    

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    return new_h, new_state

class ScopeRNNCellWrapper(tf.contrib.rnn.RNNCell):
    '''this wraps an RNN cell to make sure it uses the same scope every time
    it's called'''

    def __init__(self, cell, name):
        '''ScopeRNNCellWrapper constructor'''

        self._cell = cell
        self.scope = tf.VariableScope(None, name)

    @property
    def output_size(self):
        '''return cell output size'''

        return self._cell.output_size

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        with tf.variable_scope(self.scope):
            output, new_state = self._cell(inputs, state, scope)

        return output, new_state

class StateOutputWrapper(tf.contrib.rnn.RNNCell):
    '''this wraps an RNN cell to make it output its concatenated state instead
        of the output'''

    def __init__(self, cell):
        '''StateOutputWrapper constructor'''

        self._cell = cell

    @property
    def output_size(self):
        '''return cell output size'''

        return sum([int(x) for x in nest.flatten(self._cell.state_size)])

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        _, new_state = self._cell(inputs, state, scope)
        output = tf.concat(nest.flatten(new_state), axis=1)

        return output, new_state
