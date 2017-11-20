'''@file evaluator.py
contains the Evaluator class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from nabu.processing import input_pipeline
import pdb

class Evaluator(object):
    '''the general evaluator class

    an evaluator is used to evaluate the performance of a model'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, dataconf, model, output_name):
        '''Evaluator constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            output_name: the name of the output of the model to concider
            model: the model to be evaluated
        '''

        self.conf = conf
        self.model = model
        
        self.output_name = output_name

        #get the database configurations
        inputs = self.model.input_names
        input_sections = [conf.get('evaluator', i).split(' ') for i in inputs]
        self.input_dataconfs = []
        for sectionset in input_sections:
            self.input_dataconfs.append([])
            for section in sectionset:
                self.input_dataconfs[-1].append(dict(dataconf.items(section)))

        targets = conf.get('evaluator', 'targets').split(' ')
        if targets == ['']:
            targets = []
        target_sections = [conf.get('evaluator', o).split(' ') for o in targets]
        self.target_dataconfs = []
        for sectionset in target_sections:
            self.target_dataconfs.append([])
            for section in sectionset:
                self.target_dataconfs[-1].append(dict(dataconf.items(section)))

    def evaluate(self):
        '''evaluate the performance of the model

        Returns:
            - the loss as a scalar tensor
            - the number of batches in the validation set as an integer
        '''

        batch_size = int(self.conf.get('evaluator', 'batch_size'))
        requested_utts = int(self.conf.get('evaluator','requested_utts'))

        with tf.name_scope('evaluate'):

            #get the list of filenames fo the validation set
            data_queue_elements, _ = input_pipeline.get_filenames(
                self.input_dataconfs + self.target_dataconfs)

	    max_number_of_elements = len(data_queue_elements)
	    number_of_elements = min([max_number_of_elements,requested_utts])
	    
            #compute the number of batches in the validation set
            numbatches = number_of_elements/batch_size
            number_of_elements = numbatches*batch_size
            print '%d utterances will be used for evaluation' %(number_of_elements)

            #cut the data so it has a whole numbe of batches
            data_queue_elements = data_queue_elements[:number_of_elements]

            #create a queue to hold the filenames
            data_queue = tf.train.string_input_producer(
                string_tensor=data_queue_elements,
                shuffle=False,
                seed=None,
                capacity=batch_size*2)

            #create the input pipeline
            data, seq_length = input_pipeline.input_pipeline(
                data_queue=data_queue,
                batch_size=batch_size,
                numbuckets=1,
                dataconfs=self.input_dataconfs + self.target_dataconfs
            )

            inputs = {
                self.model.input_names[i]: d
                for i, d in enumerate(data[:len(self.input_dataconfs)])}

            seq_length = {
                self.model.input_names[i]: d
                for i, d in enumerate(seq_length[:len(self.input_dataconfs)])}

            target_names = self.conf.get('evaluator', 'targets').split(' ')
            targets = {
                target_names[i]: d
                for i, d in enumerate(data[len(self.input_dataconfs):])}

            #target_seq_length = {
                #target_names[i]: d
                #for i, d in enumerate(seq_length[len(self.input_dataconfs):])}
	    
	    outputs = self._get_outputs(inputs, seq_length)
	    outputs = outputs[self.output_name]

            loss, norm = self.compute_loss(targets, outputs, seq_length)
            self.loss = loss
            self.norm = norm
            self.outputs = outputs
            self.seq_length = seq_length
            self.inputs = inputs
            self.targets = targets

        return loss, norm, numbatches, outputs, seq_length

    @abstractmethod
    def _get_outputs(self, inputs, seq_length):
        '''compute the validation outputs for a batch of data

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x ...] tensors
            seq_length: The sequence lengths of the input utterances, this
                is a list of [batch_size] vectors

        Returns:
            the outputs'''

    
    @abstractmethod
    def compute_loss(self, targets, logits, seq_length):
        '''compute the validation loss for a batch of data

        Args:
            targets: a dictionary of [batch_size x time x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x time x ...] tensor containing
                the logits
            seq_length: a dictionary of [batch_size] vectors containing
                the sequence lengths

        Returns:
            a scalar value containing the loss'''
