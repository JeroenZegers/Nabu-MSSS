'''@file evaluator_factory.py
contains the Evaluator factory'''

from . import   task_loss_evaluator

def factory(evaluator):
    '''
    gets an evaluator class

    Args:
        evaluator: the evaluator type

    Returns:
        an evaluator class
    '''

    if evaluator == 'task_loss_evaluator':
        return task_loss_evaluator.TaskLossEvaluator
    else:
        raise Exception('Undefined evaluator type: %s' % evaluator)
