"""@file ntt_rec.py
contains de NTTRec class"""

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer
import collections

NTTState_names = ('iter_count', 'input_spec', 'input_seq_length', 'res_mask', 'all_masks')
NTTState = collections.namedtuple('NTTState', NTTState_names)
NTTOutputs = ['all_masks']


class NTTRec(model.IterableModel):
	"""A recurrent network based on NTT paper: 'Listening to each speaker one by one with recurrent selective hearing
	networks
	'"""

	def __init__(self, conf, name=None):
		super(NTTRec, self).__init__(conf, name=name)
		self.num_outputs = len(self.output_inds())

	def _get_outputs(self, inputs):
		"""
		Create the variables and do the forward computation

		Args:
			inputs: see NTTstate

		Returns:
			- output, which is a [batch_size x time x ...] tensors
		"""
		print 'Ik kan wrs gwn concat, dblstm en feedforward model hergebruiken'
		num_layers = 2
		num_units = [600, 600]
		layer_norm = False
		recurrent_dropout = 1.0
		activation_fn = tf.nn.tanh

		blstm_layers = []
		for l in range(num_layers):
			blstm_layers.append(layer.BLSTMLayer(
				num_units=num_units[l],
				layer_norm=layer_norm,
				recurrent_dropout=recurrent_dropout,
				activation_fn=activation_fn))

		next_iter_count = inputs.iter_count + 1
		input_spec = inputs.input_spec
		input_seq_length = inputs.input_seq_length
		res_mask = inputs.res_mask
		all_masks = inputs.all_masks

		logits = tf.concat([input_spec, res_mask], -1)

		for l in range(num_layers):
			logits = blstm_layers[l](logits, input_seq_length, 'layer' + str(l))

		mask = tf.contrib.layers.fully_connected(
			inputs=logits,
			num_outputs=129,
			activation_fn=tf.nn.sigmoid)

		new_res_mask = res_mask - mask
		new_res_mask = tf.clip_by_value(new_res_mask, 0, 1)

		all_masks_shape = all_masks.get_shape()
		all_masks = tf.concat(
			[all_masks[:, :, :, :inputs.iter_count], tf.expand_dims(mask, -1), all_masks[:, :, :, inputs.iter_count + 1:]],
			axis=-1)
		all_masks.set_shape(all_masks_shape)
		# all_masks=tf.scatter_update(
		# 	ref=all_masks,
		# 	indices=inputs.iter_count,
		# 	updates=mask
		# )
		# tf.assign(all_masks[inputs.iter_count], mask)
		# all_masks[inputs.iter_count] = mask

		new_ntt_state = NTTState(
			iter_count=next_iter_count, input_spec=input_spec, input_seq_length=input_seq_length, res_mask=new_res_mask,
			all_masks=all_masks)
		return new_ntt_state

	def zero_state(self, inputs, input_seq_length, is_training):
		"""
		Define a zero state to initialize the iterative process

		Args:


		Returns:

		"""

		iter_count = tf.constant(0, dtype=tf.int32)

		# code not available for multiple inputs!!
		if len(inputs) > 1:
			raise 'The implementation of NTT expects 1 input and not %d' % len(inputs)
		else:
			input_spec = inputs[0]
		if is_training and float(self.conf['input_noise']) > 0:
			input_spec = input_spec + tf.random_normal(
				tf.shape(input_spec),
				stddev=float(self.conf['input_noise']))

		input_seq_length = input_seq_length

		res_mask = tf.ones(tf.shape(input_spec))

		all_masks = tf.zeros(
			tf.concat(
				[tf.shape(input_spec), tf.constant([self.max_iters])], axis=-1))  # dim: B x T x F x nrSmax

		zero_state = NTTState(
			iter_count=iter_count, input_spec=input_spec, input_seq_length=input_seq_length, res_mask=res_mask,
			all_masks=all_masks)

		return zero_state

	def output_inds(self):

		"""
		Indices that indicate with loopvariables will be used in the output of the model

		Returns:
			indices: indicate with loopvariables will be used in the output of the model

		"""
		indices = [ind for ind, name in enumerate(NTTState_names) if name in NTTOutputs]
		return indices


