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
        
        
        
        #get the database configurations for all input-output pairs and the modelpaths. 
	#Currently only accepting 1 input per output.
	self.output_names = task_eval_conf['outputs'].split(' ')
	self.input_names=dict()
	self.input_dataconfs=[]
	self.model_paths=dict()
	for output_name in self.output_names:
	    #input config
	    input_name = 'input_%s'%output_name
	    if input_name == ['']:
		input_name = []
	    self.input_names[output_name] = input_name
	    
	    self.input_dataconfs.append(dict(dataconf.items(task_eval_conf[input_name])))
	    
	    #model paths
	    self.model_paths[output_name] = task_eval_conf['modelpath_%s'%output_name].split(' ')
	
	
	self.target_names = task_eval_conf['targets'].split(' ')
	if self.target_names == ['']:
	    self.target_names = []
	target_sections = [task_eval_conf[o] for o in self.target_names]
	self.target_dataconfs = []
	for section in target_sections:
	    self.target_dataconfs.append(dict(dataconf.items(section)))
	

    def evaluate(self):
        '''evaluate the performance of the model

        Returns:
            - the loss as a scalar tensor
            - the number of batches in the validation set as an integer
        '''

        batch_size = int(self.conf.get('evaluator', 'batch_size'))
        requested_utts = int(self.conf.get('evaluator','requested_utts'))

        with tf.name_scope('evaluate'):
	    data_queue_elements, _ = input_pipeline.get_filenames(
		self.input_dataconfs + self.target_dataconfs)
	    
	    max_number_of_elements = len(data_queue_elements)
	    number_of_elements = min([max_number_of_elements,requested_utts])
	    
	    #compute the number of batches in the validation set
	    numbatches = number_of_elements/batch_size
	    number_of_elements = numbatches*batch_size
	    print '%d utterances will be used for evaluation' %(number_of_elements)

	    #cut the data so it has a whole number of batches
	    data_queue_elements = data_queue_elements[:number_of_elements]
		
	    
	    #create the data queue and queue runners (inputs are allowed to get shuffled. I already did this so set to False)
	    data_queue = tf.train.string_input_producer(
		string_tensor=data_queue_elements,
		shuffle=False,
		seed=None,
		capacity=batch_size*2)
	  
	  
	  
	  
	    #data_queue = dict()
	    ##inputs
	    #for output_name in self.output_names:
		##get the filenames
		#data_queue_elements, _ = input_pipeline.get_filenames(
		    #self.input_dataconfs[output_name])
		
		#max_number_of_elements = len(data_queue_elements)
		#number_of_elements = min([max_number_of_elements,requested_utts])
		
		##compute the number of batches in the validation set
		#numbatches = number_of_elements/batch_size
		#number_of_elements = numbatches*batch_size
		#print '%d utterances will be used for evaluation' %(number_of_elements)

		##cut the data so it has a whole number of batches
		#data_queue_elements = data_queue_elements[:number_of_elements]
		    
		
		##create the data queue and queue runners (inputs are allowed to get shuffled. I already did this so set to False)
		#data_queue[output_name] = tf.train.string_input_producer(
		    #string_tensor=data_queue_elements,
		    #shuffle=False,
		    #seed=None,
		    #capacity=batch_size*2)
		
	    ##targets
	    ##get the filenames
	    #data_queue_elements, _ = input_pipeline.get_filenames(
		#self.target_dataconfs)
	    ##cut the data so it has a whole number of batches
	    #data_queue_elements = data_queue_elements[:number_of_elements]
		    
	    ##create the data queue and queue runners (inputs are allowed to get shuffled. I already did this so set to False)
	    #data_queue['targets'] = tf.train.string_input_producer(
		#string_tensor=data_queue_elements,
		#shuffle=False,
		#seed=None,
		#capacity=batch_size*2)	  
		
		
	    #create the input pipeline
	    data, seq_length = input_pipeline.input_pipeline(
		data_queue=data_queue,
		batch_size=batch_size,
		numbuckets=1,
		dataconfs=self.input_dataconfs + self.target_dataconfs
	    )
	
	    #split data into inputs and targets
	    inputs=dict()
	    seq_lengths=dict()
	    for ind,output_name in enumerate(self.output_names):
		inputs[output_name] = data[ind]
		seq_lengths[output_name]=seq_length[ind]
	    targets=dict()
	    for ind,target_name in enumerate(self.target_names):
		targets[target_name]=data[len(self.output_names)+ind]
	
	    #get the logits
	    logits=dict()
	    for output_name in self.output_names:
		logits[output_name] = self._get_outputs(
		    inputs={self.input_names[output_name]: inputs[output_name]},
		    seq_length={self.input_names[output_name]: seq_lengths[output_name]},
		    output_name=output_name)

	  
	    ##get the logits
            #logits=dict()
            #seq_lengths=dict()
            #for output_name in self.output_names:      
		##create the input pipeline
		#data, seq_length = input_pipeline.input_pipeline(
		    #data_queue=data_queue[output_name],
		    #batch_size=batch_size,
		    #numbuckets=1,
		    #dataconfs=self.input_dataconfs[output_name]
		#)

		#inputs = {self.input_names[output_name]: data[0]}
		#seq_length = {self.input_names[output_name]: seq_length[0]}
		#seq_lengths[output_name]=seq_length[self.input_names[output_name]]
		
		##compute the training outputs of the model
		#logits[output_name] = self._get_outputs(inputs=inputs,
							#seq_length=seq_length,
							#output_name=output_name)
		
	    
	    ##get the targets
	    ##create the input pipeline
	    #data, _ = input_pipeline.input_pipeline(
		#data_queue=data_queue['targets'],
		#batch_size=batch_size,
		#numbuckets=1,
		#dataconfs=self.target_dataconfs
	    #)

	    #targets = {
		#self.target_names[i]: d
		#for i, d in enumerate(data)}

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
