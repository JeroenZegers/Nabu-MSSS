'''@file plain_variables.py
contains de PlainVariables class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import pdb

class PlainVariables(model.Model):
    '''Returns vectors on indexing'''

    def  _get_outputs(self, inputs, input_seq_length=None, is_training=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the indexes, this is a list of
                [batch_size x nr_indeces] tensors
            input_seq_length: None
            is_training: None

        Returns:
            - outputs, which is a dictionary of [batch_size x 1 x (nr_indeces*vec_dim)]
                tensors
        '''
        
	
	#code not available for multiple inputs!!
	if len(inputs) > 1:
	    raise 'The implementation of PlainVariables expects 1 input and not %d' %len(inputs)
	else:
	    inputs=inputs[0]
	    
	with tf.variable_scope(self.scope):

	    #the complete vector set
	    vector_set = tf.get_variable('vector_set',initializer=tf.truncated_normal([int(self.conf['tot_vecs']),
				      int(self.conf['vec_dim'])], 
				      stddev=tf.sqrt(2/float(self.conf['vec_dim']))))

	    inputs = tf.expand_dims(inputs, -1)
	    
	    output = tf.gather_nd(vector_set, inputs)
	    output = tf.reshape(output,[tf.shape(output)[0],1,-1])
	    
        return output
