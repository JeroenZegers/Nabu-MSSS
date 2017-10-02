'''@file evaluator_factory.py
contains the Evaluator factory'''

from . import   loss_evaluator

def factory(evaluator):
    '''
    gets an evaluator class

    Args:
        evaluator: the evaluator type

    Returns:
        an evaluator class
    '''

    if evaluator == 'loss_evaluator':
        return loss_evaluator.LossEvaluator
    else:
        raise Exception('Undefined evaluator type: %s' % evaluator)
