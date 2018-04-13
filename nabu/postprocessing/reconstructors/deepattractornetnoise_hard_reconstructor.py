'''@file deepclustering_reconstructor.py
contains the reconstor class using deep clustering'''

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import os
import pdb

class DeepattractornoisehardReconstructor(mask_reconstructor.MaskReconstructor):
    '''the deepclustering reconstructor class

    a reconstructor using deep clustering'''
    requested_output_names = ['bin_emb','noise_filter']
    noise_threshold = 0.75

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''DeepclusteringReconstructor constructor

        Args:
        conf: the reconstructor configuration as a dictionary
        evalconf: the evaluator configuration as a ConfigParser
        dataconf: the database configuration
        rec_dir: the directory where the reconstructions will be stored'''

        super(DeepattractornoisehardReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)

        #get the usedbins reader
        usedbins_name = conf['usedbins']
        usedbins_dataconf = dict(dataconf.items(usedbins_name))
        self.usedbins_reader = data_reader.DataReader(usedbins_dataconf,self.segment_lengths)

        #directory where cluster centroids will be stored
        self.center_store_dir = os.path.join(rec_dir,'cluster_centers')
        if not os.path.isdir(self.center_store_dir):
            os.makedirs(self.center_store_dir)


    def _get_masks(self,output, utt_info):
        '''estimate the masks

        Args:
            output: the output of a single utterance of the neural network
                    tensor of dimension [Txfeature_dimension*emb_dim]

        Returns:
            the estimated masks'''

        embeddings = output['bin_emb']
        noise_filter = output['noise_filter']
        #only the non-silence bins will be used for the clustering
        usedbins, _ = self.usedbins_reader(self.pos)

        [T,F] = np.shape(usedbins)
        emb_dim = np.shape(embeddings)[1]/F
        N = T*F
        if np.shape(embeddings)[0] != T:
            raise 'Number of frames in usedbins does not match the sequence length'
        if np.shape(noise_filter)[0] != T:
            raise 'Number of frames in usedbins does not match the sequence length'
        if np.shape(noise_filter)[1] != F:
            raise 'Number of noise filter outputs does not match number of frequency bins'
        #reshape the outputs
        output = embeddings[:T,:]
        output_resh = np.reshape(output,[T*F,emb_dim])

        noise_filter_reshape = np.reshape(noise_filter[:T,:],[T*F,1])
        noise_filt_bin = np.reshape(noise_filter_reshape >self.noise_threshold,T*F)

        usedbins_resh = np.reshape(usedbins, T*F)
        filt = np.logical_and(usedbins_resh,noise_filt_bin)
        #Only keep the active bins (above threshold) for clustering
        output_speech_resh = output_resh[filt] # dim:N' x embdim (N' is number of bins that are used N'<N)
        if np.shape(output_speech_resh)[0] < 2:
	     return np.zeros([self.nrS,T,F])
        #apply kmeans clustering and assign each bin to a clustering
        kmeans_model=KMeans(n_clusters=self.nrS, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
        for _ in range(5):
            # Sometime it fails due to some indexerror and I'm not sure why. Just retry then. max 5 times
            try:
                kmeans_model.fit(output_speech_resh)
            except IndexError:
              continue
            break

        A = kmeans_model.cluster_centers_ # dim: nrS x embdim


        prod_1 = np.matmul(A,output_resh.T) # dim: nrS x N
        numerator = np.exp(prod_1-np.max(prod_1,axis=0))
        denominator = np.sum(numerator,axis=0)
        M = numerator/denominator
        M_final = np.multiply(M,np.transpose(noise_filter_reshape))

        #reconstruct the masks from the cluster labels
        masks = np.reshape(M_final,[self.nrS,T,F])
        np.save(os.path.join(self.center_store_dir,utt_info['utt_name']),kmeans_model.cluster_centers_)
        return masks
