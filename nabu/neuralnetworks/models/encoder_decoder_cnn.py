'''@file encoder_decoder_cnn.py
contains de EncoderDecoderCNN class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import numpy as np
import pdb

class EncoderDecoderCNN(model.Model):
    '''A CNN classifier with encoder-decoder shape 
    (https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)
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
        num_filters_1st_layer = int(self.conf['num_filters_1st_layer'])
        f_pool_rate = int(self.conf['f_pool_rate'])
        t_pool_rate = int(self.conf['t_pool_rate'])
        num_encoder_layers = int(self.conf['num_encoder_layers'])
        num_decoder_layers = num_encoder_layers
        num_centre_layers = int(self.conf['num_centre_layers'])
        
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
	encoder_layers = []
	for l in range(num_encoder_layers):
	    num_filters_l = num_filters_1st_layer * 2**l
	    
	    max_pool_filter = [1, 1]
	    if np.mod(l+1, t_pool_rate) == 0:
		max_pool_filter[0] = 2
	    if np.mod(l+1, f_pool_rate) == 0:
		max_pool_filter[1] = 2
		
	    encoder_layers.append(layer.Conv2D(num_filters=num_filters_l,
					       kernel_size=kernel_size,
					       strides=(1,1),
					       padding='same',
					       activation_fn=activation_fn,
					       layer_norm=layer_norm,
					       max_pool_filter=max_pool_filter))
	  
	# the centre layers
	centre_layers = []
	for l in range(num_centre_layers):
	    num_filters_l = num_filters_1st_layer * 2**num_encoder_layers
		
	    centre_layers.append(layer.Conv2D(num_filters=num_filters_l,
					       kernel_size=kernel_size,
					       strides=(1,1),
					       padding='same',
					       activation_fn=activation_fn,
					       layer_norm=layer_norm,
					       max_pool_filter=(1, 1)))
	  
	# the decoder layers
	decoder_layers = []
	for l in range(num_encoder_layers):
	    corresponding_encoder_l = num_encoder_layers-1-l
	    num_filters_l = encoder_layers[corresponding_encoder_l].num_filters
	    strides = encoder_layers[corresponding_encoder_l].max_pool_filter
	    
	    decoder_layers.append(layer.Conv2D(num_filters=num_filters_l,
					       kernel_size=kernel_size,
					       strides=strides,
					       padding='same',
					       activation_fn=activation_fn,
					       layer_norm=layer_norm,
					       max_pool_filter=(1,1),
					       transpose=True))
	  
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
	    
	    with tf.variable_scope('encoder'):
		encoder_outputs=[]
		for l in range(num_encoder_layers):
		    with tf.variable_scope('layer_%s'%l):
		      
			logits = encoder_layers[l](logits)
			
			encoder_outputs.append(logits)

			if is_training and float(self.conf['dropout']) < 1:
			    raise 'have to check wheter dropout is implemented correctly'
			    logits = tf.nn.dropout(logits, float(self.conf['dropout']))
	      
	    with tf.variable_scope('centre'):
		for l in range(num_centre_layers):
		    with tf.variable_scope('layer_%s'%l):
		      
			logits = centre_layers[l](logits)

			if is_training and float(self.conf['dropout']) < 1:
			    raise 'have to check wheter dropout is implemented correctly'
			    logits = tf.nn.dropout(logits, float(self.conf['dropout']))
	    
	    with tf.variable_scope('decoder'):
		for l in range(num_decoder_layers):
		    with tf.variable_scope('layer_%s'%l):
			corresponding_encoder_l = num_encoder_layers-1-l
			corresponding_encoder_output = encoder_outputs[corresponding_encoder_l]
			decoder_input = tf.concat([logits, corresponding_encoder_output], -1)
			logits = decoder_layers[l](decoder_input)
			
			if is_training and float(self.conf['dropout']) < 1:
			    raise 'have to check wheter dropout is implemented correctly'
			    logits = tf.nn.dropout(logits, float(self.conf['dropout']))
			
			#get wanted output size
			if corresponding_encoder_l==0:
			    wanted_size = tf.shape(inputs)
			else:
			    wanted_size = tf.shape(encoder_outputs[corresponding_encoder_l-1])
			#if corresponding_encoder_l==0:
			    #wanted_size = inputs.get_shape()
			#else:
			    #wanted_size = encoder_outputs[corresponding_encoder_l-1].get_shape()
			wanted_t_size = wanted_size[1]
			wanted_f_size = wanted_size[2]
			
			#get actual output size
			output_size = tf.shape(logits)
			#output_size = logits.get_shape()
			output_t_size = output_size[1]
			output_f_size = output_size[2]
			
			#compensate for potential mismatch, by adding duplicates
			missing_t_size = wanted_t_size-output_t_size
			missing_f_size = wanted_f_size-output_f_size
			
			last_t_slice = tf.expand_dims(logits[:,-1,:,:],1)
			duplicate_logits = tf.tile(last_t_slice,[1,missing_t_size,1,1])
			logits = tf.concat([logits, duplicate_logits], 1)
			last_f_slice = tf.expand_dims(logits[:,:,-1,:],2)
			duplicate_logits = tf.tile(last_f_slice,[1,1,missing_f_size,1])
			logits = tf.concat([logits, duplicate_logits], 2)
					    
	    output = logits


        return output
