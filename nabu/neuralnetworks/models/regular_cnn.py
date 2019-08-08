'''@file regular_cnn.py
contains de RegularCNN class

OBSOLETE: USE dcnn.py INSTEAD

'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import numpy as np
import pdb

class RegularCNN(model.Model):
    '''A CNN classifier
    '''

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
         
        kernel_size = map(int, self.conf['filters'].split(' '))
        num_filters = int(self.conf['num_filters'])
        num_layers = int(self.conf['num_layers'])
        
        layer_norm=self.conf['layer_norm'] == 'True'
        
        if 'activation_fn' in self.conf:
	  if self.conf['activation_fn'] == 'tanh':
	    activation_fn = tf.nn.tanh
	  elif self.conf['activation_fn'] == 'relu':
	    activation_fn = tf.nn.relu
	  elif self.conf['activation_fn'] == 'sigmoid':
	    activation_fn = tf.nn.sigmoid
	  else:
	    raise Exception('Undefined activation function: %s' % self.conf['activation_fn'])
	else:
	  activation_fn = tf.nn.relu
	  
	# the encoder layers
	cnn_layer=layer.Conv2D(num_filters=num_filters,
				kernel_size=kernel_size,
				strides=(1,1),
				padding='same',
				activation_fn=activation_fn,
				layer_norm=layer_norm)
	  
	#code not available for multiple inputs!!
	if len(inputs) > 1:
	    raise 'The implementation of DCNN expects 1 input and not %d' %len(inputs)
	else:
	    inputs=inputs[0]
	    
        # Convolutional layers expect input channels, making 1 here.
        inputs = tf.expand_dims(inputs, -1)
	with tf.variable_scope(self.scope):
	    if is_training and float(self.conf['input_noise']) > 0:
		inputs = inputs + tf.random_normal(
		    tf.shape(inputs),
		    stddev=float(self.conf['input_noise']))
		    
	    logits = inputs
	    

	    for l in range(num_layers):
		with tf.variable_scope('layer_%s'%l):
		  
		    logits = cnn_layer(logits)

		    if is_training and float(self.conf['dropout']) < 1:
			raise 'have to check wheter dropout is implemented correctly'
			logits = tf.nn.dropout(logits, float(self.conf['dropout']))
					    
	    output = logits

        return output
