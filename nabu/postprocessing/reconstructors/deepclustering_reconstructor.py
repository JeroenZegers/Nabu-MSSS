"""@file deepclustering_reconstructor.py
contains the reconstor class using deep clustering"""

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import itertools
import os


class DeepclusteringReconstructor(mask_reconstructor.MaskReconstructor):
	"""the deepclustering reconstructor class

	a reconstructor using deep clustering"""

	requested_output_names = ['bin_emb']

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""DeepclusteringReconstructor constructor

		Args:
			conf: the reconstructor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions will be stored
		"""

		super(DeepclusteringReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		# get the usedbins reader
		usedbins_names = conf['usedbins'].split(' ')
		usedbins_dataconfs = []
		for usedbins_name in usedbins_names:
			usedbins_dataconfs.append(dict(dataconf.items(usedbins_name)))
		self.usedbins_reader = data_reader.DataReader(usedbins_dataconfs, self.segment_lengths)

		# directory where cluster centroids will be stored
		self.center_store_dir = os.path.join(rec_dir, 'cluster_centers')
		if not os.path.isdir(self.center_store_dir):
			os.makedirs(self.center_store_dir)

		# whether output will be in [time x freq_dim*emb_dim] or
		# [time x freq_dim x emb_dim]
		self.flat = False
		if 'flat' in conf['reconstruct_type']:
			self.flat = True

	def _get_masks(self, output, utt_info):
		"""estimate the masks

		Args:
			output: the output of a single utterance of the neural network
				utt_info: some info on the utterance

		Returns:
			the estimated masks"""

		embeddings = output['bin_emb']
		# only the non-silence bins will be used for the clustering
		usedbins, _ = self.usedbins_reader(self.pos)
		[T, F] = np.shape(usedbins)
		if self.flat:
			emb_dim = np.shape(embeddings)[-1]
		else:
			emb_dim = np.shape(embeddings)[-1]/F
		if np.shape(embeddings)[0] != T:
			raise Exception('Number of frames in usedbins does not match the sequence length')
	
		# reshape the outputs
		embeddings = embeddings[:T, :]
		embeddings_resh = np.reshape(embeddings, [T*F, emb_dim])
		embeddings_resh_norm = np.linalg.norm(embeddings_resh, axis=1, keepdims=True)
		embeddings_resh = embeddings_resh/embeddings_resh_norm

		# only keep the active bins (above threshold) for clustering
		usedbins_resh = np.reshape(usedbins, T*F)
		embeddings_speech_resh = embeddings_resh[usedbins_resh]

		# apply kmeans clustering and assign each bin to a clustering
		kmeans_model = KMeans(n_clusters=self.nrS, init='k-means++', n_init=10, max_iter=100, n_jobs=7)
	
		for _ in range(5):
			# Sometime it fails due to some indexerror and I'm not sure why. Just retry then. max 5 times
			try:
				kmeans_model.fit(embeddings_speech_resh)
			except IndexError:
				continue
			break
	
		predicted_labels = kmeans_model.predict(embeddings_resh)
		predicted_labels_resh = np.reshape(predicted_labels, [T, F])

		# reconstruct the masks from the cluster labels
		masks = np.zeros([self.nrS, T, F])
		for spk in range(self.nrS):
			masks[spk, :, :] = predicted_labels_resh == spk

		# store the clusters
		np.save(os.path.join(self.center_store_dir, utt_info['utt_name']), kmeans_model.cluster_centers_)

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
		target = target['binary_targets']
		target = np.reshape(target, [T, F, self.nrS])
		target = np.transpose(target, [2, 0, 1])
		usedbins_ext = np.expand_dims(usedbins, 0)
		opt_masks = np.zeros([self.nrS, T, F])
		all_perms = list(itertools.permutations(range(self.nrS)))
		for t in range(T):
			mask_frame_t = masks[:, t, :]
			target_frame_t = target[:, t, :]
			usedbins_frame_t = usedbins_ext[:, t, :]

			perm_values = np.zeros([self.nrS])
			for perm_ind, perm in enumerate(all_perms):
				perm_value = np.sum(usedbins_frame_t * np.abs(mask_frame_t[perm, :] - target_frame_t))
				perm_values[perm_ind] = perm_value

			best_perm_ind = np.argmin(perm_values)
			best_perm = all_perms[best_perm_ind]

			opt_masks[:, t, :] = mask_frame_t[best_perm, :]

		return opt_masks
