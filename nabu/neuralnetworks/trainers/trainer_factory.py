'''@file trainer_factory.py
contains the Trainer factory'''

from . import   trainer, multi_task_trainer

def factory(train_type='single_task'):
    '''
    gets a trainer class

    Args:
        train_type: the trainer type

    Returns:
        a trainer class
    '''

    if train_type == 'single_task':
	raise 'single_task training is no longer supported. Define atleast 1 task using the multi_task_trainer'
        return trainer.Trainer
    elif train_type == 'multi_task':
        return multi_task_trainer.MultiTaskTrainer
    else:
        raise Exception('Undefined trainer type: %s' % train_type)
