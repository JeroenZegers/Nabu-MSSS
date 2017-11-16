'''@file task_loss_evaluator.py
contains the TaskLossEvaluator class'''

import tensorflow as tf
import task_evaluator
from nabu.neuralnetworks.loss_computers import loss_computer_factory
from nabu.neuralnetworks.models import run_multi_model

class TaskLossEvaluator(task_evaluator.TaskEvaluator):
    '''The TaskLossEvaluator is used to evaluate'''

    def __init__(self, conf, dataconf, models, task):
        '''TaskLossEvaluator constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            models: the models to be evaluated
            task: the name of the task being evaluated
        '''


        super(TaskLossEvaluator, self).__init__(conf, dataconf, models, task)
        self.loss_computer = loss_computer_factory.factory(
		conf.get(task,'loss_type'))(
		int(conf.get('evaluator','batch_size')))


    def _get_outputs(self, inputs, seq_lengths):
        '''compute the evaluation logits for a batch of data

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x ...] tensors
            seq_length: The sequence lengths of the input utterances, this
                is a list of [batch_size] vectors

        Returns:
            the logits'''

        with tf.name_scope('evaluate_logits'):
	    logits = run_multi_model.run_multi_model(
		models=self.models,
		model_nodes=self.model_nodes, 
		model_links=self.model_links, 
		inputs=inputs, 
		inputs_links=self.inputs_links,
		output_names=self.output_names, 
		seq_lengths=seq_lengths,
		is_training=False)

        return logits


    def compute_loss(self, targets, logits, seq_length):
        '''compute the evaluation loss for a batch of data

        Args:
            targets: a dictionary of [batch_size x time x ...] tensor containing
                the targets
            logits: a dictionary of [batch_size x time x ...] tensor containing
                the logits
            seq_length: a dictionary of [batch_size] vectors containing
                the sequence lengths

        Returns:
            the loss as a scalar'''

        with tf.name_scope('evaluate_loss'):
            loss, norm = self.loss_computer(targets, logits, seq_length)
            
        return loss, norm
