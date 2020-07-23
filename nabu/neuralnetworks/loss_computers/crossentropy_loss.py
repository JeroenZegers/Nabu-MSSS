"""@file crossentropy_loss.py
contains the CrossEntropyLoss"""

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops


class CrossEntropyLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length=None):
		"""
		Compute the loss

		Creates the operation to compute the crossentropy loss

		Args:
			targets: a dictionary of [batch_size x ... x ...] tensor containing
				the targets
			logits: a dictionary of [batch_size x ... x ...] tensors containing the logits

		Returns:
			loss: a scalar value containing the loss
			norm: a scalar value indicating how to normalize the loss
		"""
		
		if 'av_anchors_time_flag' in self.lossconf and self.lossconf['av_anchors_time_flag'] in ['true', 'True']:
			av_anchors_time_flag = True
		else:
			av_anchors_time_flag = False

		spkids = tf.squeeze(targets['spkids'], 1)
		logits = logits['spkest']

		if av_anchors_time_flag:
			logits = tf.reduce_mean(logits, 1)

		loss, norm = ops.crossentropy_loss(spkids, logits, self.batch_size)

		return loss, norm
