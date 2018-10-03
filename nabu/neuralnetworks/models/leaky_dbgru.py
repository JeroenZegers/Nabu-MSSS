'''@file dbgru.py
contains de DBGRU class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer

class LeakyDBGRU(model.Model):
    '''A deep bidirectional GRU classifier with memory leakage'''

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

        #the bgru layer
        num_units = int(self.conf['num_units'])
        if 'activation_fn' in self.conf:
	  if self.conf['activation_fn'] == 'tanh':
	    activation_fn = tf.nn.tanh
	  elif self.conf['activation_fn'] == 'relu':
	    activation_fn = tf.nn.relu
	  else:
	    raise Exception('Undefined activation function: %s' % activation_fn)
	else:
	  activation_fn = tf.nn.tanh
	  
        bgru = layer.LeakyBGRULayer(
            num_units=num_units,
            leak_factor=float(self.conf['leak_factor']),
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
	    
	    for l in range(int(self.conf['num_layers'])):
		logits = bgru(logits, input_seq_length,
			      'layer' + str(l))

		if is_training and float(self.conf['dropout']) < 1:
		    logits = tf.nn.dropout(logits, float(self.conf['dropout']))
		
	    output = logits


        return output
