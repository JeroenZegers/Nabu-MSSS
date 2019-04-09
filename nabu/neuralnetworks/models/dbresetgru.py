'''@file dbresetgru.py
contains de DBResetGRU class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import numpy as np
import pdb

class DBResetGRU(model.Model):
    '''A deep bidirectional reset GRU classifier'''

    def  _get_outputs(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            - output, which is a [batch_size x time x ...] tensors
        '''

        #the bresetgru layer
        num_units = int(self.conf['num_units'])
        t_reset = int(self.conf['t_reset'])
        if 'group_size' in self.conf:
	  group_size = int(self.conf['group_size'])
	  if np.mod(t_reset, group_size) != 0:
	      raise ValueError('t_reset should be a multiple of group_size')
	else:
	  group_size = 1
	if 'symmetric_context' in self.conf and self.conf['symmetric_context']=='True':
	  symmetric_context = True
	else:
	  symmetric_context = False
	num_replicates = int(float(t_reset)/float(group_size))
        if 'activation_fn' in self.conf:
	  if self.conf['activation_fn'] == 'tanh':
	    activation_fn = tf.nn.tanh
	  elif self.conf['activation_fn'] == 'relu':
	    activation_fn = tf.nn.relu
	  elif self.conf['activation_fn'] == 'sigmoid':
	    activation_fn = tf.nn.sigmoid
	  else:
	    raise Exception('Undefined activation function: %s' % activation_fn)
	else:
	  activation_fn = tf.nn.tanh
	
        bgru = layer.BResetGRULayer(
            num_units=num_units,
            t_reset=t_reset,
            group_size=group_size,
            symmetric_context=symmetric_context,
            activation_fn=activation_fn)
	
	#code not available for multiple inputs!!
	if len(inputs) > 1:
	    raise 'The implementation of DBGRU expects 1 input and not %d' %len(inputs)
	else:
	    inputs=inputs[0]
	    
	with tf.variable_scope(self.scope):
	    if is_training and float(self.conf['input_noise']) > 0:
		inputs = inputs + tf.random_normal(
		    tf.shape(inputs),
		    stddev=float(self.conf['input_noise']))
		    
	    logits = inputs
	    
	    #expand the dimension of inputs since the reset gru expect multistate input
	    logits_multistate_for_forward = tf.expand_dims(logits,2)
	    logits_multistate_for_forward = tf.tile(logits_multistate_for_forward, tf.constant([1, 1, num_replicates, 1]))
	    logits_multistate_for_backward = logits_multistate_for_forward
	    
	    for l in range(int(self.conf['num_layers'])):
		logits, logits_multistate_for_forward, logits_multistate_for_backward = bgru(
		  logits_multistate_for_forward, logits_multistate_for_backward, input_seq_length, 
		  'layer' + str(l))

		if is_training and float(self.conf['dropout']) < 1:
		    raise 'dropout not yet implemented for state reset gru'
		    #logits = tf.nn.dropout(logits, float(self.conf['dropout']))
		
	    output = logits


        return output
