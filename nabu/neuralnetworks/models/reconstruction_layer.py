'''@file linear.py
contains the Reconstruction_Layer class'''

import tensorflow as tf
import model


class Reconstruction_Layer(model.Model):
    '''Reconstruction layer'''

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
        if len(inputs) != 2:
            raise Exception('The implementation of Reconstruction layer expects 2 inputs and not %d' %len(inputs))
        else:
            signal = inputs[0]
            mask   = inputs[1]


        with tf.variable_scope(self.scope):
            output = tf.multiply(mask,signal)
            output = tf.where(output <= 1e-30,tf.ones_like(output)* 1e-30,output)
            nom = 10*tf.log(output)
            denom = tf.log(tf.constant(10, dtype=nom.dtype))
        return nom/denom
