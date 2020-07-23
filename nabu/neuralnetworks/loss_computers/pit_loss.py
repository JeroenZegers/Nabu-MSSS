"""@file pit_loss.py
contains the PITLoss"""

import loss_computer
from nabu.neuralnetworks.components import ops
import tensorflow as tf
import warnings


class PITLoss(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the Permudation Invariant Training loss

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

		if 'activation' in self.lossconf:
			activation = self.lossconf['activation']
		else:
			activation = 'softmax'
		if 'rescale_recs' in self.lossconf:
			rescale_recs = self.lossconf['rescale_recs'] == 'True'
		else:
			rescale_recs = False
		if 'overspeakerized' in self.lossconf:
			overspeakerized = self.lossconf['overspeakerized'] == 'True'
		else:
			overspeakerized = False
		if 'transpose_order' in self.lossconf:
			transpose_order = map(int, self.lossconf['transpose_order'].split(' '))
		else:
			transpose_order = False
		if 'no_perm' in self.lossconf:
			no_perm = self.lossconf['no_perm'] == 'True'
		else:
			no_perm = False
		if 'logits_name' in self.lossconf:
			logits_name = self.lossconf['logits_name']
		else:
			logits_name = 'bin_est'

		multi_targets = targets['multi_targets']
		mix_to_mask = targets['mix_to_mask']
		seq_length = seq_length[logits_name]
		logits = logits[logits_name]
		if transpose_order:
			logits = tf.transpose(logits, transpose_order)

		loss, norm = ops.pit_loss(
			multi_targets, logits, mix_to_mask, seq_length, self.batch_size, activation=activation,
			rescale_recs=rescale_recs, overspeakerized=overspeakerized, no_perm=no_perm)
			
		return loss, norm
	
	
class PITLossSigmoid(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""
	
	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss
	
		Creates the operation to compute the Permudation Invariant Training loss
	
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
		warnings.warn('In following versions it will be required to use the PITLoss', Warning)

		multi_targets = targets['multi_targets']
		mix_to_mask = targets['mix_to_mask']
		seq_length = seq_length['bin_est']
		logits = logits['bin_est']

		loss, norm = ops.pit_loss(multi_targets, logits, mix_to_mask, seq_length, self.batch_size, activation='sigmoid')

		return loss, norm


class PITLossSigmoidScaled(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the Permudation Invariant Training loss

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
		warnings.warn('In following versions it will be required to use the PITLoss', Warning)

		multi_targets = targets['multi_targets']
		mix_to_mask = targets['mix_to_mask']
		seq_length = seq_length['bin_est']
		logits = logits['bin_est']

		loss, norm = ops.pit_loss(
			multi_targets, logits, mix_to_mask, seq_length, self.batch_size, activation='sigmoid', rescale_recs=True)

		return loss, norm


class PITLossOverspeakerized(loss_computer.LossComputer):
	"""A loss computer that calculates the loss"""

	def __call__(self, targets, logits, seq_length):
		"""
		Compute the loss

		Creates the operation to compute the Permudation Invariant Training loss

		Args:
			targets: a dictionary of [batch_size x time x ...] tensor containing
				the targets
			logits: a dictionary of [nrS x batch_size x time x ...] tensors containing the logits
			seq_length: a dictionary of [batch_size] vectors containing
				the sequence lengths

		Returns:
			loss: a scalar value containing the loss
			norm: a scalar value indicating how to normalize the loss
		"""
		warnings.warn('In following versions it will be required to use the PITLoss', Warning)

		multi_targets = targets['multi_targets']
		mix_to_mask = targets['mix_to_mask']
		seq_length = seq_length['bin_est']
		logits = logits['bin_est']

		print 'Assuming "bin_est" is already activated with sigmoid'

		loss, norm = ops.pit_loss(
			multi_targets, logits, mix_to_mask, seq_length, self.batch_size, activation=None, rescale_recs=False,
			overspeakerized=True)

		return loss, norm

