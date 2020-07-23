"""@file oraclemask_reconstructor.py
contains the reconstor class using deep clustering"""

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import itertools
import os


class OracleMaskReconstructor(mask_reconstructor.MaskReconstructor):
	"""the oracle mask reconstructor class

	a reconstructor using the oracle, ideal masks"""

	requested_output_names = []

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""DeepclusteringReconstructor constructor

		Args:
			conf: the reconstructor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions will be stored
		"""

		super(OracleMaskReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		# get the binarytargets reader
		binarytargets_names = conf['binary_targets'].split(' ')
		binarytargets_dataconfs = []
		for binarytargets_name in binarytargets_names:
			binarytargets_dataconfs.append(dict(dataconf.items(binarytargets_name)))
		self.binarytargets_reader = data_reader.DataReader(binarytargets_dataconfs, self.segment_lengths)

	def _get_masks(self, output, utt_info):
		"""returns the ideal masks

		Args:
			output: the output of a single utterance of the neural network. Not used
				utt_info: some info on the utterance. Not used

		Returns:
			the ideal masks"""

		binary_targets, _ = self.binarytargets_reader(self.pos)
		[T, FS] = np.shape(binary_targets)
		F = FS/self.nrS
	
		# reshape the outputs
		binary_targets = binary_targets[:T, :]
		binary_targets_resh = np.reshape(binary_targets, [T, F, self.nrS])
		ideal_masks = np.transpose(binary_targets_resh, [2, 0, 1])

		return ideal_masks
