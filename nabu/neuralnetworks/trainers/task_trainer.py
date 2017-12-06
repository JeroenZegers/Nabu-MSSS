'''@file task_trainer.py
neural network trainer environment'''

import tensorflow as tf
from nabu.neuralnetworks.loss_computers import loss_computer_factory
from nabu.neuralnetworks.evaluators import evaluator_factory, loss_evaluator
from nabu.processing import input_pipeline
from nabu.neuralnetworks.models import run_multi_model
import pdb

class TaskTrainer():
    '''General class on how to train for a single task.'''
    
    def __init__(self,
		  task_name,
		  trainerconf,
		  taskconf,
		  models,
		  modelconf,
		  dataconf,
		  evaluatorconf,
		  batch_size):
	'''
        TaskTrainer constructor, gathers the dataconfigs and sets the loss_computer and
        evaluator for this task.

        Args:
	    task_name: a name for the training task
            trainerconf: the trainer config
            taskconf: the config file for each task
            models: the neural net models
            modelconf: the neural net models configuration
            dataconf: the data configuration as a ConfigParser
            evaluatorconf: the evaluator configuration for evaluating
                if None no evaluation will be done
            batch_size: the size of the batch.
        '''
        
        self.task_name = task_name
	self.trainerconf=trainerconf
	self.taskconf = taskconf
	self.models = models
	self.modelconf = modelconf
	self.evaluatorconf = evaluatorconf
	self.batch_size = batch_size
	    
	#get the database configurations for all inputs, outputs, intermediate model nodes and models. 
	self.output_names = taskconf['outputs'].split(' ')
	self.input_names = taskconf['inputs'].split(' ')
	self.model_nodes = taskconf['nodes'].split(' ')
	self.input_dataconfs=[]
	for input_name in self.input_names:
	    #input config	    
	    self.input_dataconfs.append(dict(dataconf.items(taskconf[input_name])))
	
	self.target_names = taskconf['targets'].split(' ')
	if self.target_names == ['']:
	    self.target_names = []
	self.target_dataconfs=[]
	for target_name in self.target_names:
	    #target config	    
	    self.target_dataconfs.append(dict(dataconf.items(taskconf[target_name])))
	    
	self.model_links = dict()
	self.inputs_links = dict()
	for node in self.model_nodes:
	    self.model_links[node] = taskconf['%s_model'%node]
	    self.inputs_links[node] = taskconf['%s_inputs'%node].split(' ')
	    
	#create the loss computer
	self.loss_computer = loss_computer_factory.factory(
		taskconf['loss_type'])(self.batch_size)
	
	#create valiation evaluator
	evaltype = evaluatorconf.get('evaluator', 'evaluator')
	if evaltype != 'None':
	    self.evaluator = evaluator_factory.factory(evaltype)(
		conf=evaluatorconf,
		dataconf=dataconf,
		models=self.models,
		task=task_name)
	    	
    
    def set_dataqueues(self, cluster):
	'''sets the data queues'''

	#check if running in distributed model
	if 'local' in cluster.as_dict():
	    data_queue_elements, _ = input_pipeline.get_filenames(
		    self.input_dataconfs +self.target_dataconfs)
	  
	    number_of_elements = len(data_queue_elements)
	    if 'trainset_frac' in self.taskconf:
		number_of_elements=int(float(number_of_elements)*
			    float(self.taskconf['trainset_frac']))
	    print '%d utterances will be used for training' %(number_of_elements)

	    data_queue_elements = data_queue_elements[:number_of_elements]
	  
	    #create the data queue and queue runners 
	    self.data_queue = tf.train.string_input_producer(
		string_tensor=data_queue_elements,
		shuffle=False,
		seed=None,
		capacity=self.batch_size*2,
		shared_name='data_queue_%s' %(self.task_name))
	  		
	    #compute the number of steps
	    if int(self.trainerconf['numbatches_to_aggregate']) == 0:
		num_steps = (int(self.trainerconf['num_epochs'])*
			    len(data_queue_elements)/
			    self.batch_size)
	    else:
		num_steps = (int(self.trainerconf['num_epochs'])*
			    len(data_queue_elements)/
			    (self.batch_size*
			    int(self.trainerconf['numbatches_to_aggregate'])))

	    done_ops = [tf.no_op()]

	else:
	    #get the data queue
	    self.data_queue = tf.FIFOQueue(
		capacity=self.batch_size*(num_replicas+1),
		shared_name='data_queue_%s' %(self.task_name),
		name='data_queue_%s' %(self.task_name),
		dtypes=[tf.string],
		shapes=[[]])
	
	    #get the number of steps from the parameter server
	    num_steps_queue = tf.FIFOQueue(
		capacity=num_replicas,
		dtypes=[tf.int32],
		shared_name='num_steps_queue',
		name='num_steps_queue',
		shapes=[[]]
	    )

	    #set the number of steps
	    num_steps = num_steps_queue.dequeue()
	  
	    #get the done queues
	    for i in range(num_servers):
		with tf.device('job:ps/task:%d' % i):
		    done_queue = tf.FIFOQueue(
			capacity=num_replicas,
			dtypes=[tf.bool],
			shapes=[[]],
			shared_name='done_queue%d' % i,
			name='done_queue%d' % i
		    )

		    done_ops.append(done_queue.enqueue(True))
	  
	return num_steps, done_ops
    
    def train(self, learning_rate):
	'''set the training ops for this task'''
      
	with tf.variable_scope(self.task_name):
      
	    #create the optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
            
            #create the input pipeline
	    data, seq_length = input_pipeline.input_pipeline(
		data_queue=self.data_queue,
		batch_size=self.batch_size,
		numbuckets=int(self.trainerconf['numbuckets']),
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
	    logits = run_multi_model.run_multi_model(
		models=self.models,
		model_nodes=self.model_nodes, 
		model_links=self.model_links, 
		inputs=inputs, 
		inputs_links=self.inputs_links,
		output_names=self.output_names, 
		seq_lengths=seq_lengths,
		is_training=True)

	    #TODO: The proper way to exploit data paralellism is via the 
	    #SyncReplicasOptimizer defined below. However for some reason it hangs
	    #and I have not yet found a solution for it. For the moment the gradients
	    #are accumulated in a way that does not allow data paralellism and there
	    # is no advantage on having multiple workers.
	    
	    #create an optimizer that aggregates gradients
	    #if int(conf['numbatches_to_aggregate']) > 0:
		#optimizer = tf.train.SyncReplicasOptimizer(
		    #opt=optimizer,
		    #replicas_to_aggregate=int(
			#conf['numbatches_to_aggregate'])#,
		    ##total_num_replicas=num_replicas
		    #)
		    
	    #a variable to hold the batch loss
	    self.batch_loss = tf.get_variable(
			    name='batch_loss',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_batch_loss = self.batch_loss.assign(0.0)
	    
	    #a variable to hold the batch loss norm
	    self.batch_loss_norm = tf.get_variable(
			    name='batch_loss_norm',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_batch_loss_norm = self.batch_loss_norm.assign(0.0)
	    
	    #gather all trainable parameters
	    params = tf.trainable_variables()
		  
	    #a variable to hold all the gradients
	    self.grads = [tf.get_variable(
		param.op.name, param.get_shape().as_list(),
		initializer=tf.constant_initializer(0),
		trainable=False) for param in params]

	    reset_grad = tf.variables_initializer(self.grads)
	    
	    #compute the loss
<<<<<<< HEAD
	    task_minibatch_loss, task_minibatch_loss_norm = self.loss_computer(targets, logits[self.output_name], seq_length)
=======
	    task_minibatch_loss, task_minibatch_loss_norm = self.loss_computer(
		targets, logits, seq_lengths)
>>>>>>> a85c454f6d52886bb800ddf8f62179face762f02
	    
	    task_minibatch_grads_and_vars = optimizer.compute_gradients(task_minibatch_loss)
	    
	    (task_minibatch_grads, task_vars)=zip(*task_minibatch_grads_and_vars)
	    
	    #update the batch gradients with the minibatch gradients.
	    #If a minibatchgradients is None, the loss does not depent on the specific
	    #variable(s) and it will thus not be updated
	    with tf.variable_scope('update_gradients'):
		update_gradients = [grad.assign_add(batchgrad)
			  for batchgrad, grad in zip(task_minibatch_grads,self.grads)
			  if batchgrad is not None]
	    	    
	    acc_loss  = self.batch_loss.assign_add(task_minibatch_loss)
	    acc_loss_norm  = self.batch_loss_norm.assign_add(task_minibatch_loss_norm)
	    
	    #group all the operations together that need to be executed to process 
	    #a minibatch
	    self.process_minibatch = tf.group(*(update_gradients+[acc_loss]
					 +[acc_loss_norm])
					 ,name='update_grads_loss_norm')
	    
	    #an op to reset the grads, the loss and the loss norm
	    self.reset_grad_loss_norm = tf.group(*([reset_grad,reset_batch_loss,
					     reset_batch_loss_norm])
					     ,name='reset_grad_loss_norm')
	    
	    
	    #normalize the loss
	    with tf.variable_scope('normalize_loss'):
	      self.normalized_loss = self.batch_loss/self.batch_loss_norm
	    
	    #normalize the gradients if requested.
	    with tf.variable_scope('normalize_gradients'):
		if self.trainerconf['normalize_gradients']=='True':
		    self.normalize_gradients = [grad.assign(tf.divide(grad,self.batch_loss_norm))
			      for grad in self.grads]
		else:
		    self.normalize_gradients = [grad.assign(grad)
			      for grad in self.grads]
	    
	    batch_grads_and_vars = zip(self.grads, task_vars)
	    
	    with tf.variable_scope('clip'):
		clip_value = float(self.trainerconf['clip_grad_value'])
		#clip the gradients
		batch_grads_and_vars = [(tf.clip_by_value(grad, -clip_value, clip_value), var)
			  for grad, var in batch_grads_and_vars]
	    
	    #an op to apply the accumulated gradients to the variables
	    self.apply_gradients = optimizer.apply_gradients(
                        grads_and_vars=batch_grads_and_vars,
                        name='apply_gradients')

      
      
    def evaluate_evaluator(self):
	'''set the evaluation ops for this task'''
	
	with tf.variable_scope(self.task_name):
	    #a variable to hold the validation loss
	    loss = tf.get_variable(
			    name='loss',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_loss = loss.assign(0.0)
	    
	    #a variable to hold the validation loss norm
	    loss_norm = tf.get_variable(
			    name='loss_norm',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_loss_norm = loss_norm.assign(0.0)
	  
	    #evaluate a validation batch
	    val_batch_loss, val_batch_norm, valbatches, _, _ = self.evaluator.evaluate()
	    
	    acc_loss  = loss.assign_add(val_batch_loss)
	    acc_loss_norm  = loss_norm.assign_add(val_batch_norm)
	    
	    #group all the operations together that need to be executed to process 
	    #a validation batch
	    self.process_val_batch = tf.group(*([acc_loss, acc_loss_norm])
					 ,name='update_loss')
	    
	    #an op to reset the loss and the loss norm
	    self.reset_val_loss_norm = tf.group(*([reset_loss, reset_loss_norm])
					     ,name='reset_val_loss_norm')
	    
	    #normalize the loss
	    self.val_loss_normalized = loss/loss_norm
	    
	return valbatches
	    
