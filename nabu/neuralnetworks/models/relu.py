'''@file relu.py
contains the RELU class'''

import tensorflow as tf
import model

class RELU(model.Model):
    '''One or multiple RELU layers'''

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

	#code not available for multiple inputs!!
        if len(inputs) > 1:
            raise 'The implementation of RELU expects 1 input and not %d' %len(inputs)
        else:
            inputs=inputs[0]

        with tf.variable_scope(self.scope):
            if is_training and float(self.conf['input_noise']) > 0:
                inputs = inputs + tf.random_normal(
                    tf.shape(inputs),
                    stddev=float(self.conf['input_noise']))

            logits = inputs
            for l in range(int(self.conf['num_layers'])):
                logits = tf.contrib.layers.fully_connected(
                activation_fn=tf.nn.relu,
                inputs=logits,
                num_outputs=int(self.conf['output_dims']))

            output = logits

            #dropout is not recommended
            if is_training and float(self.conf['dropout']) < 1:
                output = tf.nn.dropout(output, float(self.conf['dropout']))

            if 'last_only' in self.conf and self.conf['last_only']=='True':
                output = output[:,-1,:]
                output = tf.expand_dims(output,1)

        return output
