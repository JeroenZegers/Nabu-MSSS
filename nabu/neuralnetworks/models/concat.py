'''@file concat.py
contains the Concat class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import pdb

class Concat(model.Model):
    '''Returns a model that simpely concatenates inputs'''

    def  _get_outputs(self, inputs, input_seq_length=None, is_training=None):
        '''
        concatenate the inputs over the last dimension. If the first input has 1 dimension
        more then another input, expand the dimension of the latter.

        Args:
            inputs: the inputs to concatenate, this is a list of
                [batch_size x time x ...] tensors and/or [batch_size x ...] tensors
            input_seq_length: None
            is_training: None

        Returns:
            - outputs, the concatenated inputs
        '''
        
	nr_input = len(inputs)
	out_shape=inputs[0].get_shape()
	out_dim= len(out_shape)
	for ind in range(1,nr_inputs):
	  input_tensor=inputs[ind]
	  if out_dim-len(input_tensor.get_shape())==1:
	    input_tensor=tf.expand_dims(input_tensor,1)
	    multiplicates=np.ones(out_dim)
	    multiplicates[1]=out_shape[1]
	    input_tensor=tf.tile(input_tensor,multiplicates)
	    inputs[ind]=input_tensor
	  
	  if out_dim-len(input_tensor.get_shape())>1:
	    raise 'unexpected shape for input %d' %ind
	
	output=tf.concat(inputs,-1)
	      
        return output
