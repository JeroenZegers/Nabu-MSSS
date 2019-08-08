"""@file loss_computer.py
contains de LossComputer class"""
from abc import ABCMeta


class LossComputer(object):
	"""a general class for a loss computer """
	
	__metaclass__ = ABCMeta

	def __init__(self, lossconf, batch_size):
		"""LossComputer constructor

		Args:
			lossconf: the configuration file for the loss function
			batch_size: the size of the batch to compute the loss over
		"""
		self.lossconf = lossconf
		self.batch_size = batch_size
