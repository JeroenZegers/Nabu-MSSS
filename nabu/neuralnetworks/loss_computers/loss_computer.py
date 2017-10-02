'''@file loss_computer.py
contains de LossComputer class'''
from abc import ABCMeta

class LossComputer(object):
    '''a general class for a loss computer '''
    
    __metaclass__ = ABCMeta

    def __init__(self, batch_size):
        '''LossComputer constructor

        Args:
            batch_size: the size of the batch to compute the loss over
        '''
                   
	self.batch_size = batch_size
