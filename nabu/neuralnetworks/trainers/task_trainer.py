'''@file task_trainer.py
neural network trainer environment'''

import tensorflow as tf
from nabu.neuralnetworks.loss_computers import loss_computer_factory
from nabu.neuralnetworks.evaluators import evaluator_factory, loss_evaluator
from nabu.processing import input_pipeline

class TaskTrainer():
    '''General class on how to train for a single task.'''
    
    def __init__(self,
		  task_name,
		  trainerconf,
		  taskconf,
		  model,
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
            model: the neural net model
            modelconf: the neural net model configuration
            dataconf: the data configuration as a ConfigParser
            evaluatorconf: the evaluator configuration for evaluating
                if None no evaluation will be done
            batch_size: the size of the batch.
        '''
        
        self.task_name = task_name
	self.trainerconf=trainerconf
	self.taskconf = taskconf
	self.model = model
	self.modelconf = modelconf
	self.evaluatorconf = evaluatorconf
	self.batch_size = batch_size
	    
	#get the database configurations
	self.input_names = modelconf.get('io', 'inputs').split(' ')
	if self.input_names == ['']:
	    self.input_names = []
	input_sections = [taskconf[i].split(' ') for i in self.input_names]
	self.nr_input_sections = len(input_sections)
	
	self.input_dataconfs = []
	for sectionset in input_sections:
	    self.input_dataconfs.append([])
	    for section in sectionset:
		self.input_dataconfs[-1].append(dict(dataconf.items(section)))
	
	self.target_names = taskconf['targets'].split(' ')
	if self.target_names == ['']:
	    self.target_names = []
	target_sections = [taskconf[o].split(' ') for o in self.target_names]
	self.target_dataconfs = []
	for sectionset in target_sections:
	    self.target_dataconfs.append([])
	    for section in sectionset:
		self.target_dataconfs[-1].append(dict(dataconf.items(section)))
		
	if trainerconf['multi_trainer']=='single_1to1':
	    #the model has the same output for every task
	    self.output_name = modelconf.get('io', 'outputs')
	    
	elif trainerconf['multi_trainer']=='single_1tomany':
	    #the model has a separate output per task
	    self.output_name = taskconf['output']
	else:
	    raise 'did not understand multi_trainer style: %s' %trainerconf['multi_trainer']
		
	#create the loss computer
	self.loss_computer = loss_computer_factory.factory(
		taskconf['loss_type'])(self.batch_size)
	
	#create valiation evaluator
	evaltype = evaluatorconf.get('evaluator', 'evaluator')
	if evaltype != 'None':
	    self.evaluator = evaluator_factory.factory(evaltype)(
		conf=evaluatorconf,
		dataconf=dataconf,
		model=self.model,
		output_name=self.output_name,
		task=task_name)
	    	
    
    def set_dataqueues(self, cluster):
	'''sets the data queues'''
      
	#check if running in distributed model
	if 'local' in cluster.as_dict():

	    #get the filenames
	    data_queue_elements, _ = input_pipeline.get_filenames(
		self.input_dataconfs + self.target_dataconfs)
	    
	    #create the data queue and queue runners (inputs are allowed to get shuffled. I already did this so set to False)
	    self.data_queue = tf.train.string_input_producer(
		string_tensor=data_queue_elements,
		shuffle=False,
		seed=None,
		capacity=self.batch_size*2,
		shared_name='data_queue_' + self.task_name)
	    
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
	    with tf.device(chief_ps):

		#get the data queue
		self.data_queue = tf.FIFOQueue(
		    capacity=self.batch_size*(num_replicas+1),
		    shared_name='data_queue_' + self.task_name,
		    name='data_queue_' + self.task_name,
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

	    inputs = {
		self.input_names[i]: d
		for i, d in enumerate(data[:self.nr_input_sections])}
	    seq_length = {
		self.input_names[i]: d
		for i, d in enumerate(seq_length[:self.nr_input_sections])}
	    targets = {
		self.target_names[i]: d
		for i, d in enumerate(data[self.nr_input_sections:])}
	    #target_seq_length = {
		#self.target_names[i]: d
		#for i, d in enumerate(seq_length[self.nr_input_sections:])}

	    #compute the training outputs of the model
	    logits = self.model(
		inputs=inputs,
		input_seq_length=seq_length,
		is_training=True)


	    #TODO: The proper way to exploit data paralellism is via the 
	    #SyncReplicasOptimizer defined below. However for some reason it hangs
	    #and I have not yet found a solution for it. For the moment the gradients
	    #are accumulated in a way that does not allow data paralellism and there
	    # is no advantage on having multiple workers. (We also accumulate the loss)
	    
	    #create an optimizer that aggregates gradients
	    #if int(conf['numbatches_to_aggregate']) > 0:
		#optimizer = tf.train.SyncReplicasOptimizer(
		    #opt=optimizer,
		    #replicas_to_aggregate=int(
			#conf['numbatches_to_aggregate'])#,
		    ##total_num_replicas=num_replicas
		    #)
	    self.batch_loss = tf.get_variable(
			    name='batch_loss',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_batch_loss = self.batch_loss.assign(0.0)
	    
	    self.batch_loss_norm = tf.get_variable(
			    name='batch_loss_norm',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_batch_loss_norm = self.batch_loss_norm.assign(0.0)
	    
	    #Assuming all params are trainable for this task. TODO: fix this
	    params = tf.trainable_variables()
		  
	    self.grads = [tf.get_variable(
		param.op.name, param.get_shape().as_list(),
		initializer=tf.constant_initializer(0),
		trainable=False) for param in params]
	    
	    reset_grad = tf.variables_initializer(self.grads)
	    
	    #compute the loss
	    task_minibatch_loss, task_minibatch_loss_norm = self.loss_computer(
		targets, logits[self.output_name], seq_length)
	    
	    task_minibatch_grads_and_vars = optimizer.compute_gradients(task_minibatch_loss)
	    
	    (task_minibatch_grads, task_vars)=zip(*task_minibatch_grads_and_vars)
	    
	    with tf.variable_scope('update_gradients'):
		update_gradients = [grad.assign_add(batchgrad)
			  for batchgrad, grad in zip(task_minibatch_grads,self.grads)]
	    
	    with tf.variable_scope('normalize_loss'):
	      self.normalized_loss = self.batch_loss/self.batch_loss_norm
	    
	    acc_loss  = self.batch_loss.assign_add(task_minibatch_loss)
	    acc_loss_norm  = self.batch_loss_norm.assign_add(task_minibatch_loss_norm)
	    
	    self.process_minibatch = tf.group(*(update_gradients+[acc_loss]
					 +[acc_loss_norm])
					 ,name='update_grads_loss_norm')
	    
	    self.reset_grad_loss_norm = tf.group(*([reset_grad,reset_batch_loss,
					     reset_batch_loss_norm])
					     ,name='reset_grad_loss_norm')
	    
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
	    
	    self.apply_gradients = optimizer.apply_gradients(
                        grads_and_vars=batch_grads_and_vars,
                        name='apply_gradients')

      
      
    def evaluate_evaluator(self):
	'''set the evaluation ops for this task'''
	
	with tf.variable_scope(self.task_name):
	  
	    loss = tf.get_variable(
			    name='loss',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_loss = loss.assign(0.0)
	    
	    loss_norm = tf.get_variable(
			    name='loss_norm',
			    shape=[],
			    dtype=tf.float32,
			    initializer=tf.constant_initializer(0),
			    trainable=False)
		      
	    reset_loss_norm = loss_norm.assign(0.0)
	  
	    val_batch_loss, val_batch_norm, valbatches, _, _ = self.evaluator.evaluate()
	    
	    acc_loss  = loss.assign_add(val_batch_loss)
	    acc_loss_norm  = loss_norm.assign_add(val_batch_norm)
	    
	    self.process_val_batch = tf.group(*([acc_loss, acc_loss_norm])
					 ,name='update_loss')
	    
	    self.reset_val_loss_norm = tf.group(*([reset_loss, reset_loss_norm])
					     ,name='reset_val_loss_norm')
	    
	    self.val_loss_normalized = loss/loss_norm
	    
	return valbatches
	    
