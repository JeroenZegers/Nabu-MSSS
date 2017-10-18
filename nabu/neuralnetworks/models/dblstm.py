'''@file dblstm.py
contains de DBLSTM class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer

class DBLSTM(model.Model):
    '''A deep bidirectional LSTM classifier'''

    def  _get_outputs(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - outputs, which is a dictionary of [batch_size x time x ...]
                tensors
        '''

        #the blstm layer
        blstm = layer.BLSTMLayer(
            num_units=int(self.conf['num_units']),
            layer_norm=self.conf['layer_norm'] == 'True',
            recurrent_dropout=float(self.conf['recurrent_dropout']))

        #do the forward computation
        outputs = {}
	
	#code not available for multiple inputs!!
	if len(inputs) > 1:
	    raise 'The implementation DBLSTM expects 1 input and not %d' %len(inputs)
	  
	with tf.variable_scope(self.scope):
	    for inp in inputs:
		if is_training and float(self.conf['input_noise']) > 0:
                    inputs[inp] = inputs[inp] + tf.random_normal(
                        tf.shape(inputs[inp]),
                        stddev=float(self.conf['input_noise']))
		    
	    logits = inputs.values()[0]
	    
	    for l in range(int(self.conf['num_layers'])):
		logits = blstm(logits, input_seq_length.values()[0],
			      'layer' + str(l))

	    if is_training and float(self.conf['dropout']) < 1:
		logits = tf.nn.dropout(logits, float(self.conf['dropout']))
		
	    #if multiple outputs requested, model is completely shared, except for the
	    #output layer
	    for o in self.output_dims:
		output = tf.contrib.layers.linear(
		    inputs=logits,
		    num_outputs=self.output_dims[o],
		    scope=('outlayer_%s')%(o))

		outputs[o] = output

        return outputs
