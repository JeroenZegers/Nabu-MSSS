"""@file time_anchor_deepattractornet_softmax_reconstructor.py
contains the reconstor class using deep attractor network with softmax maskers"""

import mask_reconstructor
import numpy as np
import os


class WeightedAnchorDeepattractorSoftmaxReconstructor(mask_reconstructor.MaskReconstructor):
	"""the deepattractor softmax reconstructor class with weighted anchors

	a reconstructor using deep attractor netwerk with softmax maskers with weighted anchors"""
	requested_output_names = ['bin_emb', 'anchors', 'spk_weights']

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""WeightedAnchorDeepattractorSoftmaxReconstructor constructor

		Args:
		conf: the reconstructor configuration as a dictionary
		evalconf: the evaluator configuration as a ConfigParser
		dataconf: the database configuration
		rec_dir: the directory where the reconstructions will be stored
		task: task name
		"""

		super(WeightedAnchorDeepattractorSoftmaxReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		if 'normalize' in conf and conf['normalize'] == 'True':
			self.normalize = True
		else:
			self.normalize = False

		# directory where cluster centroids will be stored
		self.center_store_dir = os.path.join(rec_dir, 'cluster_centers')
		if not os.path.isdir(self.center_store_dir):
			os.makedirs(self.center_store_dir)

	def _get_masks(self, output, utt_info):
		"""estimate the masks

		Args:
			output: the output of a single utterance of the neural network
					tensor of dimension [Txfeature_dimension*emb_dim]

		Returns:
			the estimated masks"""

		embeddings = output['bin_emb']
		anchors = output['anchors']
		weights = output['spk_weights']

		# Get number of time frames and frequency cells
		T, out_dim = np.shape(embeddings)
		# Calculate the used embedding dimension
		emb_dim = np.shape(anchors)[-1]
		F = out_dim/emb_dim

		if np.shape(embeddings)[0] != T:
			raise Exception('Number of frames in usedbins does not match the sequence length')

		# reshape the outputs
		output = embeddings[:T, :]
		# output_resh is a N times emb_dim matrix with the embedding vectors for all cells
		output_resh = np.reshape(output, [T, F, emb_dim])
		if self.normalize:
			output_resh /= (np.linalg.norm(output_resh, axis=-1, keepdims=True) + 1e-12)

		anchors = np.expand_dims(anchors, 0)
		anchors = anchors * np.expand_dims(weights, -1)

		prod_1 = np.matmul(anchors, np.transpose(output_resh, [0, 2, 1]))  # dim: T x nrS x F
		tmp = np.exp(prod_1)
		masks = tmp / (np.sum(tmp, axis=1, keepdims=True) + 1e-12)

		# reconstruct the masks from the cluster labels
		masks = np.transpose(masks, [1, 0, 2])
		np.save(os.path.join(self.center_store_dir, utt_info['utt_name']), anchors)
		return masks
