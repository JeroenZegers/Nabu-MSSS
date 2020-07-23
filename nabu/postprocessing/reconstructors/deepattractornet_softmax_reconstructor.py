"""@file deepattractornet_softmax_reconstructor.py
contains the reconstor class using deep attractor network with softmax maskers"""

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import os
import warnings


class DeepattractorSoftmaxReconstructor(mask_reconstructor.MaskReconstructor):
    """the deepattractor softmax reconstructor class

    a reconstructor using deep attractor netwerk with softmax maskers"""
    requested_output_names = ['bin_emb']

    def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
        """DeepclusteringReconstructor constructor

        Args:
        conf: the reconstructor configuration as a dictionary
        evalconf: the evaluator configuration as a ConfigParser
        dataconf: the database configuration
        rec_dir: the directory where the reconstructions will be stored
        task: task name
        """

        warnings.warn(
            'In following versions this function will become deprecated. Use deepattractornet_reconstructor.py instead',
            Warning)

        super(DeepattractorSoftmaxReconstructor, self).__init__(
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

    def _get_masks(self, output, utt_info):
        """estimate the masks

        Args:
            output: the output of a single utterance of the neural network
                    tensor of dimension [Txfeature_dimension*emb_dim]

        Returns:
            the estimated masks"""

        embeddings = output['bin_emb']
        # only the non-silence bins will be used for the clustering
        usedbins, _ = self.usedbins_reader(self.pos)

        # Get number of time frames and frequency cells
        [T, F] = np.shape(usedbins)
        # Calculate the used embedding dimension
        emb_dim = np.shape(embeddings)[1]/F

        if np.shape(embeddings)[0] != T:
            raise Exception('Number of frames in usedbins does not match the sequence length')

        # reshape the outputs
        output = embeddings[:T, :]
        # output_resh is a N times emb_dim matrix with the embedding vectors for all cells
        output_resh = np.reshape(output, [T*F, emb_dim])

        # Only keep the active bins (above threshold) for clustering
        usedbins_resh = np.reshape(usedbins, T*F)
        output_speech_resh = output_resh[usedbins_resh]  # dim:K' x embdim (K' is number of bins that are used K'=<K)

        # apply kmeans clustering and assign each bin to a clustering
        kmeans_model = KMeans(n_clusters=self.nrS, init='k-means++', n_init=10, max_iter=100, n_jobs=7)
        for _ in range(5):
            # Sometime it fails due to some indexerror and I'm not sure why. Just retry then. max 5 times
            try:
                kmeans_model.fit(output_speech_resh)
            except IndexError:
                continue
            break

        # get cluster centers
        A = kmeans_model.cluster_centers_  # dim: nrS x embdim

        prod_1 = np.matmul(A, output_resh.T)  # dim: nrS x K
        tmp = np.exp(prod_1)
        masks = tmp / np.sum(tmp, axis=0)

        # reconstruct the masks from the cluster labels
        masks = np.reshape(masks, [self.nrS, T, F])
        np.save(os.path.join(self.center_store_dir, utt_info['utt_name']), kmeans_model.cluster_centers_)
        return masks
