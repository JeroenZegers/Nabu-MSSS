"""@file stackedmasks_noise_reconstructor.py
contains the reconstructor class using deep clustering"""

import mask_reconstructor
import numpy as np
from nabu.postprocessing import data_reader
import itertools


class StackedmasksNoiseReconstructor(mask_reconstructor.MaskReconstructor):
	"""the stacked masks reconstructor class

	a reconstructor using that uses stacked masks"""

	requested_output_names = ['bin_est', 'noise_filter']

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""StackedmasksReconstructor constructor

		Args:
			conf: the reconstructor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions will be stored
		"""

		if 'softmax_flag' in conf:
			raise Exception('Softmax argument is deprecated. Use output_activation')

		if 'output_activation' in conf:
			self.output_activation = conf['output_activation']
		else:
			self.output_activation = 'softmax'

		super(StackedmasksNoiseReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		if optimal_frame_permutation:
			# get the usedbins reader
			usedbins_names = conf['usedbins'].split(' ')
			usedbins_dataconfs = []
			for usedbins_name in usedbins_names:
				usedbins_dataconfs.append(dict(dataconf.items(usedbins_name)))
			self.usedbins_reader = data_reader.DataReader(usedbins_dataconfs, self.segment_lengths)

	def _get_masks(self, output, utt_info):
		"""get the masks by simply destacking the stacked masks into separate masks and
		normalizing them with softmax

		Args:
			output: the output of a single utterance of the neural network
				utt_info: some info on the utterance

		Returns:
			the estimated masks"""

		[T, target_dim] = np.shape(output['bin_est'])
		F = target_dim/self.nrS

		masks = np.reshape(output['bin_est'], [T, F, self.nrS], 'F')
		masks = np.transpose(masks, [2, 0, 1])

		# apply softmax

		if self.output_activation == 'softmax':
			masks = softmax(masks, axis=0)
		elif self.output_activation == 'sigmoid':
			masks = sigmoid(masks)
		elif self.output_activation in ['None', 'none', None]:
			pass
		else:
			raise Exception('Unknown requested output activation')

		noise_filters = output['noise_filter']
		noise_filters = np.expand_dims(noise_filters, 0)

		masks = masks * noise_filters

		return masks

	def _get_masks_opt_frame_perm(self, output, target, utt_info):
		"""estimate the masks

		Args:
			output: the output of a single utterance of the neural network
			target: the target of a single utterance of the neural network
			utt_info: some info on the utterance

		Returns:
			the estimated masks"""
		# only the non-silence bins will be used for determining the optimal frame level permutation
		usedbins, _ = self.usedbins_reader(self.pos)

		[T, F] = np.shape(usedbins)

		masks = self._get_masks(output, utt_info)

		# apply frame lever permutations to get the optimal masks, according to the targets
		target = target['multi_targets']
		target = np.reshape(target, [T, F, self.nrS])
		target = np.transpose(target, [2, 0, 1])
		target = np.abs(target)
		sum_targets = np.sum(target, axis=0, keepdims=True)
		irms = target/sum_targets
		usedbins_ext = np.expand_dims(usedbins, 0)
		opt_masks = np.zeros([self.nrS, T, F])
		all_perms = list(itertools.permutations(range(self.nrS)))
		for t in range(T):
			mask_frame_t = masks[:, t, :]
			irms_frame_t = irms[:, t, :]
			usedbins_frame_t = usedbins_ext[:, t, :]

			perm_values = np.zeros([self.nrS])
			for perm_ind, perm in enumerate(all_perms):
				perm_value = np.sum(usedbins_frame_t * np.abs(mask_frame_t[perm, :] - irms_frame_t))
				perm_values[perm_ind] = perm_value

			best_perm_ind = np.argmin(perm_values)
			best_perm = all_perms[best_perm_ind]

			opt_masks[:, t, :] = mask_frame_t[best_perm, :]

		return opt_masks


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def softmax(x, axis=0):
	tmp = np.exp(x)
	return tmp / np.sum(tmp, axis=axis)
