"""@file crossentropy_multi_loss.py
contains the CrossEntropyMultiLoss"""

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops


class CrossEntropyMultiLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length=None):
		"""
		Compute the loss

		Creates the operation to compute the crossentropy multi loss

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

		if 'resh_logits' in self.lossconf and self.lossconf['resh_logits'] in ['true', 'True']:
			resh_logits = True
		else:
			resh_logits = False

		if 'allow_permutation' not in self.lossconf or self.lossconf['allow_permutation'] == 'True':
			allow_permutation = True
		else:
			allow_permutation = False

		spkids = targets['spkids']
		logits = logits['spkest']

		if av_anchors_time_flag:
			logits = tf.reduce_mean(logits, 1)
		if resh_logits:
			nrS = spkids.get_shape()[1]
			logits = tf.reshape(logits, [self.batch_size, nrS, -1])

		loss, norm = ops.crossentropy_multi_loss(spkids, logits, self.batch_size, allow_permutation=allow_permutation)

		return loss, norm
