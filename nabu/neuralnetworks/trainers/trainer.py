'''@file trainer.py
neural network trainer environment'''

import os
#from abc import ABCMeta, abstractmethod, abstractproperty
import time
import cPickle as pickle
import tensorflow as tf
from nabu.processing import input_pipeline
from nabu.neuralnetworks.models import model_factory
from nabu.neuralnetworks.loss_computers import loss_computer_factory
from nabu.neuralnetworks.evaluators import evaluator_factory, loss_evaluator
from nabu.neuralnetworks.components import hooks
import pdb

class Trainer():
    '''General class outlining the training environment of a model.'''
    
    def __init__(self,
                 conf,
                 tasksconf,
                 dataconf,
                 modelconf,
                 evaluatorconf,
                 expdir,
                 init_filename,
                 server,
                 task_index):
        '''
        Trainer constructor, creates the training graph

        Args:
            conf: the trainer config
            taskconf: will be ignored in Trainer. It is used in multi_task_trainer.MultiTaskTrainer()
            dataconf: the data configuration as a ConfigParser
            modelconf: the neural net model configuration
            evaluatorconf: the evaluator configuration for evaluating
                if None no evaluation will be done
            expdir: directory where the summaries will be written
            init_filename: filename of the network that should be used to
            initialize the model. Put to None if no network is available/wanted.
            server: optional server to be used for distributed training
            task_index: optional index of the worker task in the cluster
        '''

        self.expdir = expdir
        self.server = server
        self.conf = conf
        self.task_index = task_index
        self.init_filename = init_filename
        
        self.batch_size = int(conf['batch_size'])

        cluster = tf.train.ClusterSpec(server.server_def.cluster)

        #create the graph
        self.graph = tf.Graph()

        #get the database configurations
        input_names = modelconf.get('io', 'inputs').split(' ')
        if input_names == ['']:
            input_names = []
        input_sections = [conf[i].split(' ') for i in input_names]
        input_dataconfs = []
        for sectionset in input_sections:
            input_dataconfs.append([])
            for section in sectionset:
                input_dataconfs[-1].append(dict(dataconf.items(section)))

        output_names = conf['targets'].split(' ')
        if output_names == ['']:
            output_names = []
        target_sections = [conf[o].split(' ') for o in output_names]
        target_dataconfs = []
        for sectionset in target_sections:
            target_dataconfs.append([])
            for section in sectionset:
                target_dataconfs[-1].append(dict(dataconf.items(section)))

        #create the model
        modelfile = os.path.join(expdir, 'model', 'model.pkl')
        with open(modelfile, 'wb') as fid:
            self.model = model_factory.factory(
		modelconf.get('model','architecture'))(
                conf=modelconf)
            pickle.dump(self.model, fid)

	#create the loss computer
	self.loss_computer = loss_computer_factory.factory(
		conf['loss_type'])(self.batch_size)
		
        #create the evaluator
        evaltype = evaluatorconf.get('evaluator', 'evaluator')
        if evaltype != 'None':
	    evaluator = evaluator_factory.factory(evaltype)(
		conf=evaluatorconf,
		dataconf=dataconf,
		model=self.model
	    )

        if 'local' in cluster.as_dict():
            num_replicas = 1
            device = tf.DeviceSpec(job='local')
        else:
            #distributed training
            num_replicas = len(cluster.as_dict()['worker'])
            num_servers = len(cluster.as_dict()['ps'])
            ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
                num_tasks=num_servers,
                load_fn=tf.contrib.training.byte_size_load_fn
            )
            device = tf.train.replica_device_setter(
                ps_tasks=num_servers,
                ps_strategy=ps_strategy)
            chief_ps = tf.DeviceSpec(
                job='ps',
                task=0)

        self.is_chief = task_index == 0

        #define the placeholders in the graph
        with self.graph.as_default():

            #create a local num_steps variable
            self.num_steps = tf.get_variable(
                name='num_steps',
                shape=[],
                dtype=tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False
            )

            #a variable to hold the amount of steps already taken
            self.global_step = tf.get_variable(
                name='global_step',
                shape=[],
                dtype=tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False)

            should_terminate = tf.get_variable(
                name='should_terminate',
                shape=[],
                dtype=tf.bool,
                initializer=tf.constant_initializer(False),
                trainable=False)

            self.terminate = should_terminate.assign(True).op

            #create a check if training should continue
            self.should_stop = tf.logical_or(
                tf.greater_equal(self.global_step, self.num_steps),
                should_terminate)

            with tf.device(device):

                #check if running in distributed model
                if 'local' in cluster.as_dict():

                    #get the filenames
                    data_queue_elements, _ = input_pipeline.get_filenames(
                        input_dataconfs + target_dataconfs)
		    
                    #create the data queue and queue runners (inputs get shuffled! I already did this so set to False)
                    data_queue = tf.train.string_input_producer(
                        string_tensor=data_queue_elements,
                        shuffle=False,
                        seed=None,
                        capacity=self.batch_size*2,
                        shared_name='data_queue')

                    #compute the number of steps
                    if int(conf['numbatches_to_aggregate']) == 0:
			num_steps = (int(conf['num_epochs'])*
				    len(data_queue_elements)/
				    self.batch_size)
		    else:
			num_steps = (int(conf['num_epochs'])*
				    len(data_queue_elements)/
				    (self.batch_size*
				    int(conf['numbatches_to_aggregate'])))

                    #set the number of steps
                    self.set_num_steps = self.num_steps.assign(num_steps).op
                    self.done = tf.no_op()

                else:
                    with tf.device(chief_ps):

                        #get the data queue
                        data_queue = tf.FIFOQueue(
                            capacity=self.batch_size*(num_replicas+1),
                            shared_name='data_queue',
                            name='data_queue',
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
                        self.set_num_steps = self.num_steps.assign(
                            num_steps_queue.dequeue()).op

                    #get the done queues
                    done_ops = []
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

                    self.done = tf.group(*done_ops)

                #training part
                with tf.variable_scope('train'):

                    #create the input pipeline
                    data, seq_length = input_pipeline.input_pipeline(
                        data_queue=data_queue,
                        batch_size=self.batch_size,
                        numbuckets=int(conf['numbuckets']),
                        dataconfs=input_dataconfs + target_dataconfs
                    )

                    inputs = {
                        input_names[i]: d
                        for i, d in enumerate(data[:len(input_sections)])}
                    seq_length = {
                        input_names[i]: d
                        for i, d in enumerate(seq_length[:len(input_sections)])}
                    targets = {
                        output_names[i]: d
                        for i, d in enumerate(data[len(input_sections):])}
                    #target_seq_length = {
                        #output_names[i]: d
                        #for i, d in enumerate(seq_length[len(input_sections):])}

                    #compute the training outputs of the model
                    logits = self.model(
                        inputs=inputs,
                        input_seq_length=seq_length,
                        is_training=True)

                    #a variable to scale the learning rate (used to reduce the
                    #learning rate in case validation performance drops)
                    learning_rate_fact = tf.get_variable(
                        name='learning_rate_fact',
                        shape=[],
                        initializer=tf.constant_initializer(1.0),
                        trainable=False)

                    #compute the learning rate with exponential decay and scale
                    #with the learning rate factor
                    self.learning_rate = (tf.train.exponential_decay(
                        learning_rate=float(conf['initial_learning_rate']),
                        global_step=self.global_step,
                        decay_steps=self.num_steps,
                        decay_rate=float(conf['learning_rate_decay']))
                                          * learning_rate_fact)

                    #create the optimizer
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)

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

		    total_loss = tf.get_variable(
			name='total_loss',
			shape=[],
			dtype=tf.float32,
			initializer=tf.constant_initializer(0),
			trainable=False)
		    
		    reset_loss = total_loss.assign(0.0)
		    
		    total_loss_norm = tf.get_variable(
			name='total_loss_norm',
			shape=[],
			dtype=tf.float32,
			initializer=tf.constant_initializer(0),
			trainable=False)
		    
		    reset_loss_norm = total_loss_norm.assign(0.0)
		    
		    self.normalized_loss = total_loss/total_loss_norm
		    
                    #compute the loss
                    loss, norm = self.loss_computer(
                        targets, logits, seq_length)
		    
		    acc_loss = total_loss.assign_add(loss)
		    acc_loss_norm = total_loss_norm.assign_add(norm)

                    ##compute the gradients
                    #grads_and_vars = optimizer.compute_gradients(self.loss)

                    #with tf.variable_scope('clip'):
			#clip_value = float(conf['clip_grad_value'])
                        ##clip the gradients
                        #grads_and_vars = [(tf.clip_by_value(grad, -clip_value, clip_value), var)
                                 #for grad, var in grads_and_vars]
		    
		    
	    
		    self.params = tf.trainable_variables()
		  
		    grads = [tf.get_variable(
                        param.op.name, param.get_shape().as_list(),
                        initializer=tf.constant_initializer(0),
                        trainable=False) for param in self.params]
		    
		    if 'normalize_gradients' in conf and conf['normalize_gradients'] == 'True':
			self.normalize_gradients = [grad.assign(tf.divide(grad,total_loss_norm))
					 for grad in grads]
		    else:
			self.normalize_gradients = [grad.assign(grad)
					 for grad in grads]
		    
		    reset_grad = tf.variables_initializer(grads)
		    
		    self.reset_grad_loss_norm = tf.group(*([reset_loss,reset_loss_norm,
					   reset_grad]))
	       
                    #compute the gradients
                    minibatch_grads_and_vars = optimizer.compute_gradients(loss)

                    with tf.variable_scope('clip'):
			clip_value = float(conf['clip_grad_value'])
                        #clip the gradients
                        minibatch_grads_and_vars = [(tf.clip_by_value(grad, -clip_value, clip_value), var)
                                 for grad, var in minibatch_grads_and_vars]		    
		 
		    (minibatchgrads,minibatchvars)=zip(*minibatch_grads_and_vars)
                    
                    #update gradients by accumulating them
		    update_gradients = [grad.assign_add(batchgrad)
		      for batchgrad, grad in zip(minibatchgrads,grads)]
		    
		    self.process_minibatch = tf.group(*([acc_loss,acc_loss_norm]+
					  update_gradients),
					  name='process_minibatch')
		    
                    #opperation to apply the gradients
		    grads_and_vars=list(zip(grads,minibatchvars))
                    apply_gradients_op = optimizer.apply_gradients(
                        grads_and_vars=grads_and_vars,
                        global_step=self.global_step,
                        name='apply_gradients')

                    #all remaining operations with the UPDATE_OPS GraphKeys
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    #create an operation to update the gradients, the batch_loss
                    #and do all other update ops
                    self.update_op = tf.group(
                        *([apply_gradients_op] + update_ops),
                        name='update')

                if evaltype != 'None':

                    #validation part
                    with tf.variable_scope('validate'):

                        #create a variable to save the last step where the model
                        #was validated
                        validated_step = tf.get_variable(
                            name='validated_step',
                            shape=[],
                            dtype=tf.int32,
                            initializer=tf.constant_initializer(
                                -int(conf['valid_frequency'])),
                            trainable=False)

                        #a check if validation is due
                        self.should_validate = tf.greater_equal(
                            self.global_step - validated_step,
                            int(conf['valid_frequency']))

			val_loss = tf.get_variable(
					name='loss',
					shape=[],
					dtype=tf.float32,
					initializer=tf.constant_initializer(0),
					trainable=False)
				  
			reset_val_loss = val_loss.assign(0.0)
			
			val_loss_norm = tf.get_variable(
					name='loss_norm',
					shape=[],
					dtype=tf.float32,
					initializer=tf.constant_initializer(0),
					trainable=False)
				  
			reset_val_loss_norm = val_loss_norm.assign(0.0)
	    
                        #compute the loss
                        val_batch_loss, val_batch_norm, self.valbatches, _, _ = evaluator.evaluate()
                        
                        acc_val_loss  = val_loss.assign_add(val_batch_loss)
			acc_val_loss_norm  = val_loss_norm.assign_add(val_batch_norm)

			self.process_val_batch = tf.group(*([acc_val_loss, acc_val_loss_norm])
						    ,name='process_val_batch')
			
			self.reset_val_loss_norm = tf.group(*([reset_loss, reset_loss_norm])
							,name='reset_val_loss_norm')
			
			self.validation_loss = val_loss/val_loss_norm

                        #update the learning rate factor
                        self.half_lr = learning_rate_fact.assign(
                            learning_rate_fact/2).op

                        #create an operation to updated the validated step
                        self.update_validated_step = validated_step.assign(
                            self.global_step).op

                        #variable to hold the best validation loss so far
                        self.best_validation = tf.get_variable(
                            name='best_validation',
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(1.79e+308),
                            trainable=False)

                        #op to update the best velidation loss
                        self.update_best = self.best_validation.assign(
                            self.validation_loss).op

                        #a variable that holds the amount of workers at the
                        #validation point
                        waiting_workers = tf.get_variable(
                            name='waiting_workers',
                            shape=[],
                            dtype=tf.int32,
                            initializer=tf.constant_initializer(0),
                            trainable=False)

                        #an operation to signal a waiting worker
                        self.waiting = waiting_workers.assign_add(1).op

                        #an operation to set the waiting workers to zero
                        self.reset_waiting = waiting_workers.initializer

                        #an operation to check if all workers are waiting
                        self.all_waiting = tf.equal(waiting_workers,
                                                    num_replicas-1)

                        tf.summary.scalar('validation loss',
                                          self.validation_loss)
                else:
                    self.process_val_batch = None

                tf.summary.scalar('learning rate', self.learning_rate)

                #create a histogram for all trainable parameters
                for param in tf.trainable_variables():
                    tf.summary.histogram(param.name, param)

                #create the scaffold
                self.scaffold = tf.train.Scaffold()

    def train(self):
        '''train the model'''

        #look for the master if distributed training is done
        master = self.server.target

        #start the session and standart servises
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True

	chief_only_hooks = []
	
	if self.init_filename != None:
	    init_hook = hooks.LoadAtBegin(self.init_filename,
				   self.model)
	    chief_only_hooks.append(init_hook)
	    
        #create a hook for saving the final model
        save_hook = hooks.SaveAtEnd(
            os.path.join(self.expdir, 'model', 'network.ckpt'),
            self.model)
	chief_only_hooks.append(save_hook)

        #create a hook for saving and restoring the validated model
        validation_hook = hooks.ValidationSaveHook(
            os.path.join(self.expdir, 'logdir', 'validated.ckpt'),
            self.model)
	chief_only_hooks.append(validation_hook)

        #number of times validation performance was worse
        num_tries = 0


        with self.graph.as_default():
            with tf.train.MonitoredTrainingSession(
                master=master,
                is_chief=self.is_chief,
                checkpoint_dir=os.path.join(self.expdir, 'logdir'),
                scaffold=self.scaffold,
                hooks=[hooks.StopHook(self.done)],
                chief_only_hooks=chief_only_hooks,
                config=config) as sess:

                #set the number of steps
                self.set_num_steps.run(session=sess)

                #start the training loop
                #pylint: disable=E1101
                while not (sess.should_stop() or
                           self.should_stop.eval(session=sess)):
		  
                    #check if validation is due
                    if (self.process_val_batch is not None
                            and self.should_validate.eval(session=sess)):
                        if self.is_chief:
                            print ('WORKER %d: validating model'
                                   % self.task_index)

                            #get the previous validation loss
                            prev_val_loss = self.best_validation.eval(
                                session=sess)

                            #reset the validation loss
                            self.reset_val_loss_norm.run(session=sess)
                            
			    #start time
			    start = time.time()
                    
                            #compute the validation loss
                            for _ in range(self.valbatches):
                                self.process_val_batch.run(session=sess)

                            #get the current validation loss
                            [validation_loss] = sess.run([self.validation_loss])

                            print ('WORKER %d: validation loss:%.6g,'
				    'time: %f sec' %
                                   (self.task_index, validation_loss, 
				    time.time()-start))

                            #check if the validation loss is better
                            if validation_loss >= prev_val_loss:

                                print ('WORKER %d: validation loss is worse!' %
                                       self.task_index)

                                #check how many times validation performance was
                                #worse
                                if self.conf['num_tries'] != 'None':
                                    if num_tries == int(self.conf['num_tries']):
                                        validation_hook.restore()
                                        print ('WORKER %d: terminating training'
                                               % self.task_index)
                                        self.terminate.run(session=sess)
                                        break

                                num_tries += 1

                                if self.conf['go_back'] == 'True':

                                    #wait untill all workers are at validation
                                    #point
                                    while not self.all_waiting.eval(
                                            session=sess):
                                        time.sleep(1)
                                    self.reset_waiting.run(session=sess)

                                    print ('WORKER %d: loading previous model'
                                           % self.task_index)

                                    #load the previous model
                                    validation_hook.restore()
                                else:
                                    self.update_validated_step.run(session=sess)


                                if self.conf['valid_adapt'] == 'True':
                                    print ('WORKER %d: halving learning rate'
                                           % self.task_index)
                                    self.half_lr.run(session=sess)
                                    validation_hook.save()

                            else:
                                if self.conf['reset_tries'] == 'True':
                                    num_tries = 0

                                #set the validated step
                                self.update_validated_step.run(session=sess)
                                self.update_best.run(session=sess)
                                self.reset_waiting.run(session=sess)

                                #store the validated model
                                validation_hook.save()

                        else:
                            if (self.conf['go_back'] == 'True'
                                    and self.process_val_batch is not None):
                                self.waiting.run(session=sess)
                                while (self.should_validate.eval(session=sess)
                                       and not
                                       self.should_stop.eval(session=sess)):
                                    time.sleep(1)

                                if self.should_stop.eval(session=sess):
                                    break
		    
                    #start time
                    start = time.time()
                    
                    #reset the gradients for the next step
		    sess.run(fetches=[self.reset_grad_loss_norm])

		    #First, accumulate the gradients
		    for _ in range(int(self.conf['numbatches_to_aggregate'])):
			sess.run(fetches=[self.process_minibatch])
			
		    #Then, normalize the gradients
		    _ = sess.run([self.normalize_gradients])
			    
		    #Finally, apply the gradients
		    _, loss, lr, global_step, num_steps = sess.run(
			fetches=[self.update_op,
				self.normalized_loss,
				self.learning_rate,
				self.global_step,
				self.num_steps])
				

                    print(('WORKER %d: step %d/%d loss: %.6g, learning rate: %f, '
                           'time: %f sec')
                          %(self.task_index,
                            global_step,
                            num_steps,
                            loss, lr, time.time()-start))


class ParameterServer(object):
    '''a class for parameter servers'''

    def __init__(self,
                 conf,
                 modelconf,
                 dataconf,
                 server,
                 task_index):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            conf: the trainer config
            modelconf: the model configuration
            dataconf: the data configuration as a ConfigParser
            server: optional server to be used for distributed training
            task_index: optional index of the worker task in the cluster
        '''

        self.graph = tf.Graph()
        self.server = server
        self.task_index = task_index
        self.batch_size = int(conf['batch_size'])

        #distributed training
        cluster = tf.train.ClusterSpec(server.server_def.cluster)
        num_replicas = len(cluster.as_dict()['worker'])

        with self.graph.as_default():

            #the chief parameter server should create the data queue
            if task_index == 0:
                #get the database configurations
                inputs = modelconf.get('io', 'inputs').split(' ')
                if inputs == ['']:
                    inputs = []
                input_sections = [conf[i].split(' ') for i in inputs]
                input_dataconfs = []
		for sectionset in input_sections:
		    input_dataconfs.append([])
		    for section in sectionset:
			input_dataconfs[-1].append(dict(dataconf.items(section)))
                output_names = conf['targets'].split(' ')
		if output_names == ['']:
		    output_names = []
		target_sections = [conf[o].split(' ') for o in output_names]
		target_dataconfs = []
		for sectionset in target_sections:
		    target_dataconfs.append([])
		    for section in sectionset:
			target_dataconfs[-1].append(dict(dataconf.items(section)))

                data_queue_elements, _ = input_pipeline.get_filenames(
                    input_dataconfs + target_dataconfs)

                tf.train.string_input_producer(
                    string_tensor=data_queue_elements,
                    shuffle=True,
                    seed=None,
                    capacity=self.batch_size*(num_replicas+1),
                    shared_name='data_queue')
		if int(conf['numbatches_to_aggregate'])==0:
		    num_steps = (int(conf['num_epochs'])*len(data_queue_elements)/
				self.batch_size)
		else:
		    num_steps = (int(conf['num_epochs'])*len(data_queue_elements)/
				(self.batch_size*int(conf['numbatches_to_aggregate'])))

                #create a queue to communicate the number of steps
                num_steps_queue = tf.FIFOQueue(
                    capacity=num_replicas,
                    dtypes=[tf.int32],
                    shapes=[[]],
                    shared_name='num_steps_queue',
                    name='num_steps_queue'
                )

                self.set_num_steps = num_steps_queue.enqueue_many(
                    tf.constant([num_steps]*num_replicas)
                )

                #create a queue for the workers to signiy that they are done
                done_queue = tf.FIFOQueue(
                    capacity=num_replicas,
                    dtypes=[tf.bool],
                    shapes=[[]],
                    shared_name='done_queue%d' % task_index,
                    name='done_queue%d' % task_index
                )

                self.wait_op = done_queue.dequeue_many(num_replicas).op

            self.scaffold = tf.train.Scaffold()

    def join(self):
        '''wait for the workers to finish'''

        with self.graph.as_default():
            with tf.train.MonitoredTrainingSession(
                master=self.server.target,
                is_chief=False,
                scaffold=self.scaffold) as sess:

                if self.task_index == 0:
                    self.set_num_steps.run(session=sess)

                self.wait_op.run(session=sess)
