'''@file task_evaluator.py
contains the TaskEvaluator class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from nabu.processing import input_pipeline
import pdb

class TaskEvaluator(object):
    '''the general evaluator class

    an evaluator is used to evaluate the performance of a model'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, dataconf, models,  task):
        '''TaskEvaluator constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            models: the models to be evaluated
        '''

        self.conf = conf
        self.models = models
        self.task = task
        task_eval_conf = dict(conf.items(task))
        
        if 'requested_utts' in task_eval_conf:
	    self.requested_utts = int(task_eval_conf['requested_utts'])
	else:
	    self.requested_utts = int(self.conf.get('evaluator','requested_utts'))
	if 'batch_size' in task_eval_conf:
	    self.batch_size = int(task_eval_conf['batch_size'])
	else:
	    self.batch_size = int(self.conf.get('evaluator','batch_size'))
        
	#get the database configurations for all inputs, outputs, intermediate model nodes and models. 
	self.output_names = task_eval_conf['outputs'].split(' ')
	self.input_names = task_eval_conf['inputs'].split(' ')
	self.model_nodes = task_eval_conf['nodes'].split(' ')
	self.input_dataconfs=[]
	for input_name in self.input_names:
	    #input config	    
	    self.input_dataconfs.append(dict(dataconf.items(task_eval_conf[input_name])))
	
	self.target_names = task_eval_conf['targets'].split(' ')
	if self.target_names == ['']:
	    self.target_names = []
	self.target_dataconfs=[]
	for target_name in self.target_names:
	    #target config	    
	    self.target_dataconfs.append(dict(dataconf.items(task_eval_conf[target_name])))
	    
	self.model_links = dict()
	self.inputs_links = dict()
	for node in self.model_nodes:
	    self.model_links[node] = task_eval_conf['%s_model'%node]
	    self.inputs_links[node] = task_eval_conf['%s_inputs'%node].split(' ')        

    def evaluate(self):
        '''evaluate the performance of the model

        Returns:
            - the loss as a scalar tensor
            - the number of batches in the validation set as an integer
        '''


        with tf.name_scope('evaluate'):
	    data_queue_elements, _ = input_pipeline.get_filenames(
		self.input_dataconfs + self.target_dataconfs)
	    
	    max_number_of_elements = len(data_queue_elements)
	    number_of_elements = min([max_number_of_elements,self.requested_utts])
	    
	    #compute the number of batches in the validation set
	    numbatches = number_of_elements/self.batch_size
	    number_of_elements = numbatches*self.batch_size
	    print '%d utterances will be used for evaluation' %(number_of_elements)

	    #cut the data so it has a whole number of batches
	    data_queue_elements = data_queue_elements[:number_of_elements]
		
	    

	    #create the data queue and queue runners (inputs are allowed to get shuffled. I already did this so set to False)
	    data_queue = tf.train.string_input_producer(
		string_tensor=data_queue_elements,
		shuffle=False,
		seed=None,
		capacity=self.batch_size*2)
		
	    #create the input pipeline
	    data, seq_length = input_pipeline.input_pipeline(
		data_queue=data_queue,
		batch_size=self.batch_size,
		numbuckets=1,
		dataconfs=self.input_dataconfs + self.target_dataconfs
	    )
	
	    #split data into inputs and targets
	    inputs=dict()
	    seq_lengths=dict()
	    for ind,input_name in enumerate(self.input_names):
		inputs[input_name] = data[ind]
		seq_lengths[input_name] = seq_length[ind]
		    
	    targets=dict()
	    for ind,target_name in enumerate(self.target_names):
		targets[target_name]=data[len(self.input_names)+ind]

	    #get the logits
	    logits = self._get_outputs(
		inputs=inputs, 
		seq_lengths=seq_lengths)
	    
            loss, norm = self.compute_loss(targets, logits, seq_lengths)

        return loss, norm, numbatches, logits, seq_lengths


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
