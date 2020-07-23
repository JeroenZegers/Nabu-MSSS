'''@file multi_averager.py
contains de MultiAverager class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import pdb

class MultiAverager(model.Model):
    '''Initial goal: input is [batch_size x nr_samples x feat_dim]
		     classlabels is [batch_size x nr_samples x nrClasses]
		     output is [batch_size x nrClasses x feat_dim], with output
		      being the average of the samples beloing to the class
	current implementation: input is [batch_size x nr_frams x feat_dim*emb_dim]
		     classlabels is [batch_size x nr_frams x feat_dim*nrClasses]
		     output is [batch_size x nrClasses x emb_dim]
		    '''

    def  _get_outputs(self, inputs, input_seq_length, is_training=None):
        '''
        Create the variables and index them

        Args:
            inputs: xx
            input_seq_length: None
            is_training: None

        Returns:
            - outputs, which is a [batch_size x 1 x (nr_indeces*vec_dim)]
                tensor
        '''
        
	raise BaseException('Dont know where this is for, but should be renamed or put under a different model')
	if len(inputs) != 3:
	    raise 'The implementation of MultiAverager expects 3 inputs and not %d' %len(inputs)
	else:
	    logits=inputs[0]
	    classlabels=inputs[1]
	    usedbins=inputs[2]

	batch_size=logits.get_shape()[0]    
	feat_dim = usedbins.get_shape()[2]
        output_dim = logits.get_shape()[2]
        emb_dim = output_dim/feat_dim
        target_dim = classlabels.get_shape()[2]
        nrS = target_dim/feat_dim
			    
	
	V=tf.reshape(logits,[batch_size,-1,emb_dim],name='V') 
	Vnorm=tf.nn.l2_normalize(V, dim=2, epsilon=1e-12, name='Vnorm')
	
	ubresh=tf.reshape(usedbins,[batch_size,-1,1],name='ubresh')
	Y=tf.reshape(classlabels,[batch_size,-1,nrS])
	Y=tf.to_float(tf.multiply(Y,ubresh),name='Y')
	
	Ycnt=tf.reduce_sum(Y,1)
	Ycnt=tf.expand_dims(Ycnt,-1)+1e-12
	sum_s=tf.matmul(Y,Vnorm,transpose_a=True)
	output=tf.divide(sum_s,Ycnt)
	    
    
        return output
