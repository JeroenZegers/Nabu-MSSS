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
        xxx

        Args:
	    task_name: a name for the training task
            trainerconf: the trainer config
            taskconf: the config file for each task
            dataconf: the data configuration as a ConfigParser
            modelconf: the neural net model configuration
            dataconf: the data configuration as a ConfigParser
            evaluatorconf: the evaluator configuration for evaluating
                if None no evaluation will be done
            batch_size: the size of the batch. moet zelfde zijn voor elke taak in huidige manier implementatie multi_task_trainer
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
	
	self.output_names = taskconf['targets'].split(' ')
	if self.output_names == ['']:
	    self.output_names = []
	target_sections = [taskconf[o].split(' ') for o in self.output_names]
	self.target_dataconfs = []
	for sectionset in target_sections:
	    self.target_dataconfs.append([])
	    for section in sectionset:
		self.target_dataconfs[-1].append(dict(dataconf.items(section)))
		
	#create the loss computer
	self.loss_computer = loss_computer_factory.factory(
		taskconf['loss_type'])(self.batch_size)
	
	evaltype = evaluatorconf.get('evaluator', 'evaluator')
	if evaltype != 'None':
	    self.evaluator = evaluator_factory.factory(evaltype)(
		conf=evaluatorconf,
		dataconf=dataconf,
		model=self.model,
		task=task_name)
	    	
    
    def set_dataqueues(self, cluster):
      
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
    
    def get_minibatch_loss(self):
      
	with tf.variable_scope(self.task_name):
      
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
		self.output_names[i]: d
		for i, d in enumerate(data[self.nr_input_sections:])}
	    #target_seq_length = {
		#self.output_names[i]: d
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

	
	    #compute the loss
	    task_loss = self.loss_computer(
		targets, logits, seq_length)
	    
	return task_loss
      
      
    def evaluate_evaluator(self):
	
	with tf.variable_scope(self.task_name):
	  
	    val_batch_loss, valbatches, _, _ = self.evaluator.evaluate()
	    
	    return val_batch_loss, valbatches