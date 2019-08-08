"""@file deepclusteringnoise_reconstructor.py
contains the reconstor class using deep clustering for modified noise architecture"""

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import os


class DeepclusteringnoiseReconstructor(mask_reconstructor.MaskReconstructor):
	"""the deepclusteringnoise reconstructor class for modified architecture for noise

	a reconstructor using deep clustering"""

	requested_output_names = ['bin_emb', 'noise_filter']

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""DeepclusteringnoiseReconstructor constructor

		Args:
			conf: the reconstructor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions will be stored
			task: name of task
		"""

		super(DeepclusteringnoiseReconstructor, self).__init__(
			conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		if 'noise_threshold_for_kmeans' in conf:
			self.noise_threshold = float(conf['noise_threshold_for_kmeans'])
		else:
			self.noise_threshold = 0.75
		if 'min_kmeans_perc' in conf:
			self.min_kmeans_perc = float(conf['min_kmeans_perc'])
		else:
			self.min_kmeans_perc = 0.05

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

	def _get_masks(self, output, utt_info):
		"""estimate the masks

		Args:
			output: the output of a single utterance of the neural network
				utt_info: some info on the utterance

		Returns:
			the estimated masks"""

		embeddings = output['bin_emb']  # Embeddingvectors
		noise_filter = output['noise_filter']  # noise filter output network (alpha)
		# only the non-silence bins will be used for the clustering
		usedbins, _ = self.usedbins_reader(self.pos)

		[T, F] = np.shape(usedbins)
		emb_dim = np.shape(embeddings)[1]/F

		if np.shape(embeddings)[0] != T:
			raise Exception('Number of frames in usedbins does not match the sequence length')
		if np.shape(noise_filter)[0] != T:
			raise Exception('Number of frames in noise detect does not match the sequence length')
		if np.shape(noise_filter)[1] != F:
			raise Exception('Number of noise detect outputs does not match number of frequency bins')

		# reshape the embeddings vectors
		embeddings = embeddings[:T, :]
		embeddings_resh = np.reshape(embeddings, [T*F, emb_dim])
		embeddings_resh_norm = np.linalg.norm(embeddings_resh, axis=1, keepdims=True)
		embeddings_resh = embeddings_resh/embeddings_resh_norm

		if np.isnan(embeddings_resh).any():
			print 'Embedding reshape contains NaN'

		# reshape noise filter
		noise_filter = noise_filter[:T, :]
		noise_filter_resh = np.reshape(noise_filter, T*F)
		# which cells have not too much noise
		no_noise = noise_filter_resh > self.noise_threshold
		# only keep the active bins (above threshold) for clustering and not too noisy
		usedbins_resh = np.reshape(usedbins, T*F)
		filt = np.logical_and(usedbins_resh, no_noise)
		perc_for_kmeans = float(np.sum(filt))/float(np.sum(usedbins_resh))
		if perc_for_kmeans < self.min_kmeans_perc:
			print \
				'Found that less then %.1f%% (%.1f%%)of the tf bins with energy where considered non-noise for the Kmeans. ' \
				'Lowering the noise threshold so that %.1f%% of the bins will be considered' % \
				(self.min_kmeans_perc*100, perc_for_kmeans*100, self.min_kmeans_perc*100)
			num_bins_wanted = int(np.ceil(np.sum(usedbins_resh)*self.min_kmeans_perc))
			noise_filt_used_bin = noise_filter_resh * usedbins_resh
			sorted_noise_filt_used_bin_inds = np.argsort(noise_filt_used_bin)
			sorted_noise_filt_used_bin_inds = sorted_noise_filt_used_bin_inds[::-1]
			filt = sorted_noise_filt_used_bin_inds[:num_bins_wanted]

		embeddings_speech_resh = embeddings_resh[filt]
		if np.isnan(embeddings_speech_resh).any():
			print 'embeddings_speech_resh contains NaN'
		if np.shape(embeddings_speech_resh)[0] < 2:
			return np.zeros([self.nrS, T, F])
		# apply kmeans clustering and assign each bin to a cluster
		kmeans_model = KMeans(n_clusters=self.nrS, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)

		# Sometime it fails due to some indexerror and I'm not sure why. Just retry then. max 5 times
		for _ in range(5):
			try:
				kmeans_model.fit(embeddings_speech_resh)
			except IndexError:
				continue
			break
		# assign each cell to cluster
		predicted_labels = kmeans_model.predict(embeddings_resh)
		predicted_labels_resh = np.reshape(predicted_labels, [T, F])

		# reconstruct the masks from the cluster labels
		masks = np.zeros([self.nrS, T, F])
		for spk in range(self.nrS):
			masks[spk, :, :] = (predicted_labels_resh == spk)*noise_filter
		if np.isnan(masks).any():
			print 'masks contains NaN'

		# store the clusters
		np.save(os.path.join(self.center_store_dir, utt_info['utt_name']), kmeans_model.cluster_centers_)

		return masks
