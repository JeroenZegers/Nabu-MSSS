'''@file deepclusteringnoise_reconstructor.py
contains the reconstor class using deep clustering'''

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import os
import pdb

class DeepclusteringnoiseReconstructor(mask_reconstructor.MaskReconstructor):
    '''the deepclustering reconstructor class

    a reconstructor using deep clustering'''
    
    requested_output_names = ['bin_emb','noise_labels']
    noise_threshold = 0.5

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''DeepclusteringReconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions will be stored
        '''
        
        super(DeepclusteringnoiseReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)
        
        #get the usedbins reader
        usedbins_name = conf['usedbins']
        usedbins_dataconf = dict(dataconf.items(usedbins_name))
        self.usedbins_reader = data_reader.DataReader(usedbins_dataconf,self.segment_lengths)
        
        #directory where cluster centroids will be stored
        self.center_store_dir = os.path.join(rec_dir,'cluster_centers')
        if not os.path.isdir(self.center_store_dir):
	      os.makedirs(self.center_store_dir)
        

    def _get_masks(self, output, utt_info):
	'''estimate the masks

	Args:
	    output: the output of a single utterance of the neural network
            utt_info: some info on the utterance

	Returns:
	    the estimated masks'''
 
        embeddings = output['bin_emb']
        noise_detect = output['noise_labels']
        #only the non-silence bins will be used for the clustering    
        usedbins, _ = self.usedbins_reader(self.pos)
	
        [T,F] = np.shape(usedbins)
        emb_dim = np.shape(embeddings)[1]/F
	
        if np.shape(embeddings)[0] != T:
            raise 'Number of frames in usedbins does not match the sequence length'
        if np.shape(noise_detect)[0] != T:
            raise 'Number of frames in noise detect does not match the sequence length'
        if np.shape(noise_detect)[1] != F:
            raise 'Number of noise detect outputs does not match number of frequency bins'
            
	
        #reshape the outputs
        embeddings = embeddings[:T,:]
        embeddings_resh = np.reshape(embeddings,[T*F,emb_dim])
        embeddings_resh_norm = np.linalg.norm(embeddings_resh,axis=1,keepdims=True)
        embeddings_resh = embeddings_resh/embeddings_resh_norm
	    
	noise_detect = noise_detect[:T,:]
	noise_detect_resh = np.reshape(noise_detect,T*F)
	no_noise = noise_detect_resh < self.noise_threshold
        #only keep the active bins (above threshold) for clustering
        usedbins_resh = np.reshape(usedbins, T*F)
        filt = np.logical_and(usedbins_resh,no_noise)
        embeddings_speech_resh = embeddings_resh[filt]
        if np.shape(embeddings_speech_resh)[0] < 2:
            return np.zeros([self.nrS,T,F])
        #apply kmeans clustering and assign each bin to a clustering
        kmeans_model=KMeans(n_clusters=self.nrS, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
	
        for _ in range(5):
        # Sometime it fails due to some indexerror and I'm not sure why. Just retry then. max 5 times
            try:
                 kmeans_model.fit(embeddings_speech_resh)
            except IndexError:
                continue
            break
	
        predicted_labels = kmeans_model.predict(embeddings_resh)
        predicted_labels_resh = np.reshape(predicted_labels,[T,F])
	
        #reconstruct the masks from the cluster labels
        masks = np.zeros([self.nrS,T,F])
        for spk in range(self.nrS):
            masks[spk,:,:] = np.logical_and(predicted_labels_resh==spk,noise_detect<self.noise_threshold)
            
        #store the clusters
        np.save(os.path.join(self.center_store_dir,utt_info['utt_name']),kmeans_model.cluster_centers_)



        return masks
