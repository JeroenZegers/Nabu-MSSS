"""@file time_anchor_deepattractornet_softmax_reconstructor.py
contains the reconstor class using deep attractor network with softmax maskers"""

import mask_reconstructor
import numpy as np
import os


class TimeAnchorDeepattractorSoftmaxReconstructor(mask_reconstructor.MaskReconstructor):
	"""the deepattractor softmax reconstructor class with time-dependent anchors

	a reconstructor using deep attractor netwerk with softmax maskers with time-dependent anchors"""
	requested_output_names = ['bin_emb', 'anchors']

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""TimeAnchorDeepattractorSoftmaxReconstructor constructor

		Args:
		conf: the reconstructor configuration as a dictionary
		evalconf: the evaluator configuration as a ConfigParser
		dataconf: the database configuration
		rec_dir: the directory where the reconstructions will be stored
		task: task name
		"""

		super(TimeAnchorDeepattractorSoftmaxReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		if 'normalize' in conf and conf['normalize'] == 'True':
			self.normalize = True
		else:
			self.normalize = False

		if 'normalize_anchor' in conf and conf['normalize_anchor'] == 'True':
			self.normalize_anchor = True
		else:
			self.normalize_anchor = False

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
		if self.normalize_anchor:
			anchors /= (np.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-12)

		prod_1 = np.matmul(anchors, np.transpose(output_resh, [0, 2, 1]))  # dim: T x nrS x F
		tmp = np.exp(prod_1)
		masks = tmp / (np.sum(tmp, axis=1, keepdims=True) + 1e-12)

		# reconstruct the masks from the cluster labels
		masks = np.transpose(masks, [1, 0, 2])
		np.save(os.path.join(self.center_store_dir, utt_info['utt_name']), anchors)
		return masks


class TimeAnchorScalarDeepattractorSoftmaxReconstructor(mask_reconstructor.MaskReconstructor):
	"""the deepattractor softmax reconstructor class with time-dependent anchors

	a reconstructor using deep attractor netwerk with softmax maskers with time-dependent anchors"""
	requested_output_names = ['bin_emb', 'anchors', 'anchors_scale']

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""TimeAnchorScalarDeepattractorSoftmaxReconstructor constructor

		Args:
		conf: the reconstructor configuration as a dictionary
		evalconf: the evaluator configuration as a ConfigParser
		dataconf: the database configuration
		rec_dir: the directory where the reconstructions will be stored
		task: task name
		"""

		super(TimeAnchorScalarDeepattractorSoftmaxReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		if 'normalize' in conf and conf['normalize'] == 'True':
			self.normalize = True
		else:
			self.normalize = False

		if 'normalize_anchor' in conf and conf['normalize_anchor'] == 'True':
			self.normalize_anchor = True
		else:
			self.normalize_anchor = False

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
		anchors_scale = output['anchors_scale'][0, 0]

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
		if self.normalize_anchor:
			anchors /= (np.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-12)
		anchors *= anchors_scale

		prod_1 = np.matmul(anchors, np.transpose(output_resh, [0, 2, 1]))  # dim: T x nrS x F
		tmp = np.exp(prod_1)
		masks = tmp / (np.sum(tmp, axis=1, keepdims=True) + 1e-12)

		# reconstruct the masks from the cluster labels
		masks = np.transpose(masks, [1, 0, 2])
		np.save(os.path.join(self.center_store_dir, utt_info['utt_name']), anchors)
		return masks


class TimeAnchorSpkWeightsDeepattractorSoftmaxReconstructor(mask_reconstructor.MaskReconstructor):
	"""the deepattractor softmax reconstructor class with time-dependent anchors

	a reconstructor using deep attractor netwerk with softmax maskers with time-dependent anchors"""
	requested_output_names = ['bin_emb', 'anchors', 'speaker_logits']

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""TimeAnchorScalarDeepattractorSoftmaxReconstructor constructor

		Args:
		conf: the reconstructor configuration as a dictionary
		evalconf: the evaluator configuration as a ConfigParser
		dataconf: the database configuration
		rec_dir: the directory where the reconstructions will be stored
		task: task name
		"""

		super(TimeAnchorSpkWeightsDeepattractorSoftmaxReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		if 'normalize' in conf and conf['normalize'] == 'True':
			self.normalize = True
		else:
			self.normalize = False

		if 'normalize_anchor' in conf and conf['normalize_anchor'] == 'True':
			self.normalize_anchor = True
		else:
			self.normalize_anchor = False

		if 'av_speaker_logits_time_flag' in conf and conf['av_speaker_logits_time_flag'] == 'True':
			self.av_speaker_logits_time_flag = True
		else:
			self.av_speaker_logits_time_flag = False

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
		speaker_logits = output['speaker_logits']
		speaker_logits = np.expand_dims(speaker_logits, -1)

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
		if self.normalize_anchor:
			anchors /= (np.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-12)
		if self.av_speaker_logits_time_flag:
			speaker_logits = np.mean(speaker_logits, 0, keepdims=True)
		anchors *= speaker_logits

		prod_1 = np.matmul(anchors, np.transpose(output_resh, [0, 2, 1]))  # dim: T x nrS x F
		tmp = np.exp(prod_1)
		masks = tmp / (np.sum(tmp, axis=1, keepdims=True) + 1e-12)

		# reconstruct the masks from the cluster labels
		masks = np.transpose(masks, [1, 0, 2])
		np.save(os.path.join(self.center_store_dir, utt_info['utt_name']), anchors)
		return masks

