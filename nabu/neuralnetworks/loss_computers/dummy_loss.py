"""@file dummy_loss.py
contains the DummyLoss"""

import tensorflow as tf
import loss_computer


class DummyLoss(loss_computer.LossComputer):
	"""A dummy loss computer that returns zero loss and norm 1"""

	def __call__(self, targets, logits, seq_length):
		"""

		Args:
			targets: a dictionary of [batch_size x time x ...] tensor containing
				the targets
			logits: a dictionary of [batch_size x time x ...] tensors containing the logits
			seq_length: a dictionary of [batch_size] vectors containing
				the sequence lengths

		Returns:
			loss: a scalar value containing the loss
			norm: a scalar value indicating how to normalize the loss
		"""

		return tf.constant(0.0), tf.constant(1.0)
