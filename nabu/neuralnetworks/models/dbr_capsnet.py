'''@file dbr_capsnet.py
contains the DBRCapsNet class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer, ops
import pdb

class DBRCapsNet(model.Model):
    '''A capsule network with bidirectional recurrency'''

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
        if 'rec_only_vote' in self.conf and self.conf['rec_only_vote']=='True':
	    rec_only_vote = True
	else:
	    rec_only_vote = False
        if 'recurrent_probability_fn' in self.conf:
	  if self.conf['recurrent_probability_fn'] == 'sigmoid':
	    recurrent_probability_fn = tf.nn.sigmoid
	  elif self.conf['recurrent_probability_fn'] == 'unit':
	    recurrent_probability_fn = ops.unit_activation
	else:
	    recurrent_probability_fn = None
	    
        if 'accumulate_input_logits' in self.conf and self.conf['accumulate_input_logits']=='False':
	    accumulate_input_logits = False
	else:
	    accumulate_input_logits = True
	    
        if 'accumulate_state_logits' in self.conf and self.conf['accumulate_state_logits']=='False':
	    accumulate_state_logits = False
	else:
	    accumulate_state_logits = True
	    
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
	    
	    #Primary capsule.
	    with tf.variable_scope('primary_capsule'):
		output = tf.identity(inputs, 'inputs')
		input_seq_length = tf.identity(input_seq_length, 'input_seq_length')
		
		#First layer is simple bidirectional rnn layer, without activation (squash activation
		#will be applied later)
		primary_output_dim = num_capsules*capsule_dim
		primary_capsules_layer = layer.BRNNLayer(num_units=primary_output_dim, 
					   linear_out_flag=True)
		
		primary_capsules = primary_capsules_layer(output, input_seq_length)
		primary_capsules = tf.reshape(
		    primary_capsules,
		    [output.shape[0].value,
		    tf.shape(output)[1],
		    num_capsules*2,
		    capsule_dim]
		)

		primary_capsules = ops.squash(primary_capsules)

		output = tf.identity(primary_capsules, 'primary_capsules')
	      
	    # non-primary capsules
	    for l in range(1, int(self.conf['num_layers'])):
		with tf.variable_scope('layer%d' % l):
		    #a capsule layer
		    caps_brnn_layer = layer.BRCapsuleLayer(num_capsules=num_capsules, 
					      capsule_dim=capsule_dim,
					      routing_iters=routing_iters,
					      recurrent_probability_fn=recurrent_probability_fn,
					      rec_only_vote=rec_only_vote,
					      logits_prior=logits_prior,
					      accumulate_input_logits=accumulate_input_logits,
					      accumulate_state_logits=accumulate_state_logits)
		    
		    output = caps_brnn_layer(output, input_seq_length)

		    if is_training and float(self.conf['dropout']) < 1:
			output = tf.nn.dropout(output, float(self.conf['dropout']))
	
	    output_dim = num_capsules*2*capsule_dim
	    output = tf.reshape(
		output,
		[output.shape[0].value,
		tf.shape(output)[1],
		output_dim]
	    )

        return output
