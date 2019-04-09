'''@file capsnet.py
contains the CapsNet class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer, ops
import pdb

class CapsNet(model.Model):
    '''A capsule network'''

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

        num_capsules = int(self.conf['num_capsules'])
        capsule_dim=int(self.conf['capsule_dim'])
        routing_iters=int(self.conf['routing_iters'])    
        if 'logits_prior' in self.conf and self.conf['logits_prior']=='True':
	    logits_prior = True
	else:
	    logits_prior = False
	
	#code not available for multiple inputs!!
	if len(inputs) > 1:
	    raise 'The implementation of CapsNet expects 1 input and not %d' %len(inputs)
	else:
	    inputs=inputs[0]
	    
	with tf.variable_scope(self.scope):
	    if is_training and float(self.conf['input_noise']) > 0:
		inputs = inputs + tf.random_normal(
		    tf.shape(inputs),
		    stddev=float(self.conf['input_noise']))
	    
	    #Primary capsule
	    with tf.variable_scope('primary_capsule'):
		output = tf.identity(inputs, 'inputs')
		input_seq_length = tf.identity(input_seq_length, 'input_seq_length')

		output_dim = num_capsules*capsule_dim
		primary_capsules = tf.layers.dense(
		    output,
		    output_dim,
		    use_bias=False
		)
		primary_capsules = tf.reshape(
		    primary_capsules,
		    [output.shape[0].value,
		    tf.shape(output)[1],
		    num_capsules,
		    capsule_dim]
		)

		primary_capsules = ops.squash(primary_capsules)
		#prim_norm = ops.safe_norm(primary_capsules)

		#tf.add_to_collection('image', tf.expand_dims(prim_norm, 3))
		output = tf.identity(primary_capsules, 'primary_capsules')
	      
	    # non-primary capsules
	    for l in range(1, int(self.conf['num_layers'])):
		with tf.variable_scope('layer%d' % l):
		    #a capsule layer
		    caps_layer = layer.Capsule(num_capsules=num_capsules, 
					      capsule_dim=capsule_dim,
					      logits_prior=logits_prior,
					      routing_iters=routing_iters)
		    
		    output = caps_layer(output)

		    if is_training and float(self.conf['dropout']) < 1:
			output = tf.nn.dropout(output, float(self.conf['dropout']))
	
	    
	    output = tf.reshape(
		output,
		[output.shape[0].value,
		tf.shape(output)[1],
		output_dim]
	    )

        return output
