'''@file framer.py
contains the Framer and DeframerSelect class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import numpy as np
import pdb

class Framer(model.Model):
    '''Returns a model that frames the input. Assuming frames will be used in 
    bidirectional RNN'''
    def __init__(self, conf, name=None):
        '''Framer constructor

        Args:
            conf: The model configuration as a configparser object
        '''
        self.context_length = int(conf['context_length'])
        self.frame_length = 2*self.context_length+1
        super(Framer, self).__init__(conf=conf,name=name)	

    def  _get_outputs(self, inputs, input_seq_length=None, is_training=None):
        '''
 

        Args:
            inputs: the inputs to concatenate, this is a list of
                [batch_size x time x ...] tensors and/or [batch_size x ...] tensors
            input_seq_length: None
            is_training: None

        Returns:
            - outputs, the framed inputs
        '''

	nr_inputs = len(inputs)
	#code not (yet) available for multiple inputs!!
	if nr_inputs > 1:
	    raise 'The implementation of CapsNet expects 1 input and not %d' %nr_inputs
	else:
	    inputs=inputs[0]
	    
	old_batch_size = inputs.get_shape()[0]
	T = tf.shape(inputs)[1]
	out_dim = inputs.get_shape()[2]
	
	pad_block_shape = tf.TensorShape(np.concatenate([[old_batch_size],[self.context_length],[out_dim]]))
	pad_block = tf.zeros(pad_block_shape,dtype=tf.float32)
	
	new_inputs = tf.concat([pad_block, inputs, pad_block], axis=1)
	
	numframes = T

	indices = tf.tile(tf.expand_dims(tf.range(0, self.frame_length),0), (numframes, 1)) \
	  + tf.tile(tf.expand_dims(tf.range(0, numframes),-1), \
	    (1, self.frame_length))
	
	indices=tf.expand_dims(tf.expand_dims(indices,1),0)
	indices=tf.tile(indices,[old_batch_size,1,out_dim,1])
		
	ra1=tf.range(old_batch_size)
	ra1=tf.expand_dims(tf.expand_dims(tf.expand_dims(ra1,-1),-1),-1)
	ra1=tf.tile(ra1,[1,numframes,out_dim,self.frame_length])
	ra2=tf.range(out_dim)
	ra2=tf.expand_dims(tf.expand_dims(tf.expand_dims(ra2,-1),0),0)
	ra2=tf.tile(ra2,[old_batch_size,numframes,1,self.frame_length])
	
	gather_indices=tf.stack([ra1,indices,ra2],axis=-1)
 
	frames = tf.gather_nd(new_inputs,gather_indices)
	
	frames = tf.transpose(frames,[0,1,3,2])
	
	new_batch_size = old_batch_size * numframes
	frames = tf.reshape(frames, [new_batch_size, self.frame_length, out_dim])

	      
        return frames

class DeframerSelect(model.Model):
    '''Returns a model that frames the input. Assuming frames will be used in 
    bidirectional RNN'''
    def __init__(self, conf, name=None):
        '''Framer constructor

        Args:
            conf: The model configuration as a configparser object
        '''
        self.context_length = int(conf['context_length'])
        self.select_index = self.context_length
        super(DeframerSelect, self).__init__(conf=conf,name=name)
    
    def  _get_outputs(self, inputs, input_seq_length=None, is_training=None):
        '''
 

        Args:
            inputs: the inputs to concatenate, this is a list of
                [batch_size x time x ...] tensors and/or [batch_size x ...] tensors
            input_seq_length: None
            is_training: None

        Returns:
            - outputs, the deframed inputs
        '''
	nr_inputs = len(inputs)
	#code not (yet) available for multiple inputs!!
	if nr_inputs > 1:
	    raise 'The implementation of CapsNet expects 1 input and not %d' %nr_inputs
	else:
	    inputs=inputs[0]

	frame_length = inputs.get_shape()[1]
	out_dim = inputs.get_shape()[2]
	old_batch_size = input_seq_length.get_shape()[0]
	
	new_inputs = tf.reshape(inputs,[old_batch_size, -1, frame_length, out_dim])
	selected_input = new_inputs[:,:,self.select_index,:]
	
	return selected_input