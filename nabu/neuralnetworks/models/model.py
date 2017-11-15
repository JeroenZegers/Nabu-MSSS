'''@file model.py
contains de Model class'''
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import pdb

class Model(object):
    '''a general class for a deep learning model'''
    __metaclass__ = ABCMeta
    
    def __init__(self, conf, name=None):
        '''Model constructor

        Args:
            conf: The model configuration as a configparser object
        '''
        
        self.conf = conf
        
        self.scope = tf.VariableScope(False, name or type(self).__name__)


    def __call__(self, inputs, input_seq_length, is_training):

        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - output logits, which is a dictionary of [batch_size x time x ...]
                tensors
            - the output logits sequence lengths which is a dictionary of
                [batch_size] vectors
        '''

        #compute the output logits
        logits = self._get_outputs(
            inputs=inputs,
            input_seq_length=input_seq_length,
            is_training=is_training)

	self.scope.reuse_variables()
	
        return logits

    @property
    def variables(self):
        '''get a list of the models's variables'''

        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.scope.name)

    @abstractmethod
    def _get_outputs(self, inputs, input_seq_length, is_training):

        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - output logits, which is a dictionary of [batch_size x time x ...]
                tensors
        '''

        
