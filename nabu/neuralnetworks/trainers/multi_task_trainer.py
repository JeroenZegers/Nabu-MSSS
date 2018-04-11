'''@file multi_task_trainer.py
neural network trainer environment'''

import os
#from abc import ABCMeta, abstractmethod, abstractproperty
import time
import cPickle as pickle
import numpy as np
import tensorflow as tf
from nabu.neuralnetworks.models import model_factory
from nabu.neuralnetworks.components import hooks
from nabu.neuralnetworks.trainers import task_trainer as task_trainer_script
import pdb

class MultiTaskTrainer():
    '''General class outlining the multi task training environment of a model.'''

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
        MultiTaskTrainer constructor, creates the training graph

        Args:
            conf: the trainer config
            taskconf: the config file for each task
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
        self.tasksconf = tasksconf
        self.task_index = task_index
        self.init_filename = init_filename
        
        self.batch_size = int(conf['batch_size'])

        cluster = tf.train.ClusterSpec(server.server_def.cluster)

        #create the graph
        self.graph = tf.Graph()
        
         #create the model
        modelfile = os.path.join(expdir, 'model', 'model.pkl')
        model_names = modelconf.get('hyper','model_names').split(' ')
        self.models = dict()
        with open(modelfile, 'wb') as fid:
	    for model_name in model_names:		
		self.models[model_name]=model_factory.factory(
		    modelconf.get(model_name,'architecture'))(
		    conf=dict(modelconf.items(model_name)),
		    name=model_name)
            pickle.dump(self.models, fid)    
            
        evaltype = evaluatorconf.get('evaluator', 'evaluator')   

        #define a trainer per traintask
        self.task_trainers=[]
        for task in self.conf['tasks'].split(' '):
	    taskconf = self.tasksconf[task]
	    
	    task_trainer=task_trainer_script.TaskTrainer(task,conf,taskconf,self.models,modelconf,
					   dataconf,evaluatorconf,self.batch_size)
	    
	    self.task_trainers.append(task_trainer)


		
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
		num_steps = []
		done_ops = []

		#set the dataqueues for each trainer
		for task_trainer in self.task_trainers:
		  
		    task_num_steps, task_done_ops = task_trainer.set_dataqueues(cluster)
		    
		    num_steps.append(task_num_steps)
		    done_ops += task_done_ops

		self.set_num_steps = self.num_steps.assign(min(num_steps)).op
		self.done = tf.group(*done_ops)
	    	    
		#training part
                with tf.variable_scope('train'):

		    
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
		    
		    #For each task, set the task specific training ops
		    for task_trainer in self.task_trainers:
		      
			task_trainer.train(self.learning_rate)
		    
		    #Group ops over tasks
		    self.process_minibatch = tf.group(*([task_trainer.process_minibatch
					for task_trainer in self.task_trainers]),
					name='process_minibatch_all_tasks')
		    
		    self.reset_grad_loss_norm = tf.group(*([task_trainer.reset_grad_loss_norm
					    for task_trainer in self.task_trainers]),
					    name='reset_grad_loss_norm_all_tasks')
		    
		    tmp=[]
		    for task in self.task_trainers:
			tmp += task_trainer.normalize_gradients
		    self.normalize_gradients = tf.group(*(tmp), 
				     name='normalize_gradients_all_tasks')
		    
		    #accumulate losses from tasks
		    with tf.variable_scope('accumulate_losses_from_tasks'):		
			self.loss_all_tasks = [task_trainer.normalized_loss for task_trainer in self.task_trainers]			
			self.total_loss = tf.reduce_mean(self.loss_all_tasks, name='acc_loss')
		    
		    tmp=[]
		    for task_trainer in self.task_trainers:
			tmp.append(task_trainer.apply_gradients)
		    
                    #all remaining operations with the UPDATE_OPS GraphKeys
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    
                    #an op to increment the global step
                    global_step_inc = self.global_step.assign_add(1)

                    #create an operation to update the gradients, the batch_loss
                    #and do all other update ops
                    #self.update_op = tf.group(
                        #*(tmp + update_ops + [global_step_inc]),
                        #name='update')
                        
                    self.other_update_op = tf.group(
                        *(update_ops + [global_step_inc]),
                        name='other_update')
	    
		
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
	    
			#For each task, if requested, set the task specific validation ops
			#The number of validation batches is the minimum number of validation 
			#batches over all tasks.
			tasks_excluded_for_val=['None']
			if evaluatorconf.has_option('evaluator','tasks_excluded_for_val'):
			    tasks_excluded_for_val=evaluatorconf.get('evaluator','tasks_excluded_for_val').split(' ')
			self.val_task_trainers=[task_trainer for task_trainer in self.task_trainers 
			  if task_trainer.task_name not in tasks_excluded_for_val]
			
			valbatches = []
			for task_trainer in self.val_task_trainers:			  
			    valbatches.append(task_trainer.evaluate_evaluator())
			self.valbatches = min(valbatches)
			
			#Group ops over tasks
			self.process_val_batch = tf.group(*([task_trainer.process_val_batch
					for task_trainer in self.val_task_trainers]))
			
			self.reset_val_loss_norm = tf.group(*([task_trainer.reset_val_loss_norm
					for task_trainer in self.val_task_trainers]))

			self.val_loss_all_tasks=[]
			for task_trainer in self.val_task_trainers:
			    self.val_loss_all_tasks.append(task_trainer.val_loss_normalized)
			self.validation_loss = tf.reduce_mean(self.val_loss_all_tasks)
			
                        #update the learning rate factor
                        self.half_lr = learning_rate_fact.assign(
                            learning_rate_fact/2).op

                        #create an operation to updated the validated step
                        self.update_validated_step = validated_step.assign(
                            self.global_step).op

                        #variable to hold the best validation loss so far
                        self.best_validation_all_tasks=[tf.get_variable(
                            name='best_validation_task_%i'%ind,
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(1.79e+308),
                            trainable=False) for ind in range(len(self.val_task_trainers))]

                        #op to update the best validation loss
                        self.update_best_all_tasks = \
			  [best_val_task.assign(self.val_loss_all_tasks[ind]) 
			    for ind,best_val_task in enumerate(self.best_validation_all_tasks)] 

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

        #start the session and standard services
        config = tf.ConfigProto(device_count = {'CPU': 1})
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True

	chief_only_hooks = []
	
	if self.init_filename != None:
	    init_hook = hooks.LoadAtBegin(self.init_filename,
				   self.models)
	    chief_only_hooks.append(init_hook)
	    
        #create a hook for saving the final model
        save_hook = hooks.SaveAtEnd(
            os.path.join(self.expdir, 'model', 'network.ckpt'),
            self.models)
	chief_only_hooks.append(save_hook)

        #create a hook for saving and restoring the validated model
        validation_hook = hooks.ValidationSaveHook(
            os.path.join(self.expdir, 'logdir', 'validated.ckpt'),
            self.models)
	chief_only_hooks.append(validation_hook)

        #number of times validation performance was worse
        num_tries = np.zeros(len(self.val_task_trainers))
        
        #determine al parameters
        all_params=[]
	for ind,_ in enumerate(self.task_trainers):
	    all_params+=self.task_trainers[ind].params
	all_params=list(set(all_params))
	all_params=sorted(all_params,key=lambda par:par.name)

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
                
                #print the params that will be updated
                print 'parameters that will be trained:'
                for ind, param in enumerate(all_params):
		    print 'param ind %i: %s'%(ind, param.name)
		
                #start the training loop
                #pylint: disable=E1101
                while not (sess.should_stop() or
                           self.should_stop.eval(session=sess)):
		  
		    ##Validation part
                    #check if validation is due
                    if (self.process_val_batch is not None
                            and self.should_validate.eval(session=sess)):
                        if self.is_chief:
                            print ('WORKER %d: validating model'
                                   % self.task_index)

                            #get the previous validation loss for each validation task
                            prev_val_loss_all_tasks = sess.run(self.best_validation_all_tasks)

                            #reset the validation loss
                            self.reset_val_loss_norm.run(session=sess)
                            
			    #start time
			    start = time.time()
                    
                            #compute the validation loss
                            for _ in range(self.valbatches):
                                self.process_val_batch.run(session=sess)

                            #get the current validation loss
                            [validation_loss,val_loss_all_tasks] = sess.run(
			      [self.validation_loss,self.val_loss_all_tasks])

			    print_str=('WORKER %d: validation loss:%.6g,'
				    'time: %f sec' %
                                   (self.task_index, validation_loss, 
				    time.time()-start))
			    #if multiple tasks, also print individual task losses
			    if len(val_loss_all_tasks)>1:
				for ind,loss_task in enumerate(val_loss_all_tasks):
				    print_str+=(', task_loss %s: %.6g' 
				    %(self.task_trainers[ind].task_name,loss_task))
                            print print_str

                            #check if the validation loss is better, for every task
			    terminate_train=False
			    restore_validation=False
			    continue_validation=True
			    do_halve_lr=False
                            for task_ind,val_task in enumerate(self.val_task_trainers):
				if val_loss_all_tasks[task_ind] >= prev_val_loss_all_tasks[task_ind]:
				    print ('WORKER %d: validation loss is worse for %s!' %
					  (self.task_index,val_task.task_name))

				    #check how many times validation performance was
				    #worse
				    num_tries[task_ind] += 1
				    if self.conf['num_tries'] != 'None':
					if num_tries[task_ind] == int(self.conf['num_tries']):
					    terminate_train=True
				    
				    if self.conf['go_back'] == 'True':
					continue_validation=False
					restore_validation=True
				    else:
					continue_validation=True

				    if self.conf['valid_adapt'] == 'True':
					do_halve_lr=True

				else:
				    sess.run(self.update_best_all_tasks[task_ind])
				    if self.conf['reset_tries'] == 'True':
					num_tries[task_ind] = 0
					
			    #decide what to do for training based on the above task validations
			    if terminate_train:
				validation_hook.restore()
				print ('WORKER %d: terminating training'
				      % self.task_index)
				self.terminate.run(session=sess)
				break
				
			    if restore_validation:
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
				
			    if continue_validation:
				self.update_validated_step.run(session=sess)
				
			    if do_halve_lr:
				print ('WORKER %d: halving learning rate'
				      % self.task_index)
				self.half_lr.run(session=sess)
				validation_hook.save()
				
			    #
			    if np.sum(num_tries)==0:
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
		    
		    ##Training part
                    #start time
                    start = time.time()

		    #reset the gradients for the next step
		    sess.run(fetches=[self.reset_grad_loss_norm])
		    
		    old_param_values=sess.run(all_params)
		    
		    #First, accumulate the gradients
		    for _ in range(int(self.conf['numbatches_to_aggregate'])):
		      	_= sess.run([self.process_minibatch])

			#_, batch_loss, batch_loss_norm = sess.run(fetches=[self.process_minibatch,
					  #self.task_trainers[0].batch_loss,
					  #self.task_trainers[0].batch_loss_norm])
			#print (('batchloss: %.6g, batch_loss_norm: %.6g, batch_normalized_loss: %.6g')
			      #%(batch_loss,batch_loss_norm,batch_loss/(batch_loss_norm+1e-20)))
			    
		    #Then, normalize the gradients
		    _ = sess.run([self.normalize_gradients])
		    
		    #Finally, apply the gradients for each task optimizer. Get the variable values before
		    #and after the update, so stepsizes for each task can be displayed.
		    old_task_param_values=[]
		    new_task_param_values=[]
		    task_params_diff=[]
		    loss_all_tasks=[]

		    for ind,task_trainer in enumerate(self.task_trainers):
			#get the variable values before update
			if ind==0:
			  old_task_param_values.append(old_param_values)
			else:
			  old_task_param_values.append(new_task_param_values[ind-1])
			  
			#Apply the gradients in the task optimizer and get the task loss. If it is the last
			#task, also get some other stuff
			if ind+1<len(self.task_trainers):
			    [_,task_loss]=sess.run([task_trainer.apply_gradients,task_trainer.normalized_loss])
			else:
			    _, _, task_loss, lr, global_step, num_steps = \
				sess.run(
				fetches=[task_trainer.apply_gradients,
					self.other_update_op,
					task_trainer.normalized_loss,
					self.learning_rate,
					self.global_step,
					self.num_steps])
			loss_all_tasks.append(task_loss)
			#get the variable values after update
			new_task_param_values.append(sess.run(all_params))
			
			#Calculate the stepsize for each variable by calculating the difference between old
			#and new variable values. Average this per variable type (eg weights layer 1) and average.
			#Also multiply with 10000 (this is just for printing format purposes)
			task_params_diff.append([10000.0*np.mean(np.abs(new_task_param_values[ind][par_ind]-old_task_param_values[ind][par_ind]))
			  for par_ind in range(len(new_task_param_values[ind]))])

		    #Calculate loss and step size over all task optimizations
		    loss=np.mean(loss_all_tasks)
		    new_param_values=new_task_param_values[-1]
		    params_diff=[10000.0*np.mean(np.abs(new_param_values[ind]-old_param_values[ind]))
			for ind in range(len(new_param_values))]
		    
		    #_, loss,loss_all_tasks, lr, global_step, num_steps,new_param_values = sess.run(
			#fetches=[self.update_op,
				#self.total_loss,
				#self.loss_all_tasks,
				#self.learning_rate,
				#self.global_step,
				#self.num_steps,
				#all_params])

		    ##Output prompt
		    #Start the printing string with most important information
		    print_str=(('WORKER %d: step %d/%d loss: %.6g, learning rate: %f, '
                           'time: %.2f sec')
                          %(self.task_index,
                            global_step,
                            num_steps,
                            loss, lr, time.time()-start))
			  
		    #if multiple tasks, also print individual task losses
		    if len(loss_all_tasks)>1:
			print_str+=' ('
			for ind,loss_task in enumerate(loss_all_tasks):
			    print_str+=('%s: %.6g. ' 
			    %(self.task_trainers[ind].task_name,loss_task))
			print_str+=')'		    
		
		    if 'print_var_updates' in self.conf and self.conf['print_var_updates']=='True':
			#print the average variable step size
			print_str+='\n Av param upd (*10000): %.3f'%np.mean(np.array(params_diff))
			#if multiple tasks, also print individual task average variable step size
			if len(task_params_diff)>1:
			    print_str+=' ('
			    for ind,task_param_diff in enumerate(task_params_diff):
				print_str+='%s: %.3f; '%(self.task_trainers[ind].task_name,np.mean(np.array(task_param_diff)))
			    print_str+=')'
			    
			#For each variable type (eg weights layer 1) print the average step size	    
			print_str+=' ('
			for par_ind,param in enumerate(all_params):
			    if par_ind>0:
			      print_str+=';'
			    print_str+=('%i: %.3f ' %(par_ind,params_diff[par_ind]))
			    #if multiple tasks, also print for each variable type the individual task average step size
			    if len(task_params_diff)>1:
				print_str+='{'
				for ind,task_param_diff in enumerate(task_params_diff):
				    if ind>0:
				      print_str+='+'
				    print_str+=('%.3f' %(task_param_diff[par_ind]))
				print_str+='} '
			print_str+=')'
		    		
		    #print the complete string
                    print(print_str)

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
	
	raise 'class parameterserver has not yet been adapted to the multi taks trainer'
	
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
