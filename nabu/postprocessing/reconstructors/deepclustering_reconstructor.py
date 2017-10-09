'''@file deepclustering_reconstructor.py
contains the reconstor class using deep clustering'''

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import pdb

class DeepclusteringReconstructor(mask_reconstructor.MaskReconstructor):
    '''the deepclustering reconstructor class

    a reconstructor using deep clustering'''

    def __init__(self, conf, dataconf, expdir):
        '''DeepclusteringReconstructor constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configurationn
            expdir: the experiment directory
        '''
        
        super(DeepclusteringReconstructor, self).__init__(conf, dataconf, expdir)
        
        #get the usedbins reader
        usedbins_name = conf.get('reconstructor','usedbins')
        usedbins_dataconf = dict(dataconf.items(usedbins_name))
        self.usedbins_reader = data_reader.DataReader(usedbins_dataconf,self.segment_lengths)
        

    def _get_masks(self, output):
	'''estimate the masks

	Args:
	    output: the output of a single utterance of the neural network

	Returns:
	    the estimated masks'''
 
	#only the non-silence bins will be used for the clustering    
	usedbins, _ = self.usedbins_reader(self.pos)
	
	[T,F] = np.shape(usedbins)
	emb_dim = np.shape(output)[1]/F
	
	if np.shape(output)[0] != T:
	    raise 'Number of frames in usedbins does not match the sequence length'
	
	#reshape the outputs
	output = output[:T,:]
	output_resh = np.reshape(output,[T*F,emb_dim])
	output_resh_norm = np.linalg.norm(output_resh,axis=1,keepdims=True)
	output_resh = output_resh/output_resh_norm
	
	#only keep the active bins (above threshold) for clustering
	usedbins_resh = np.reshape(usedbins, T*F)
	output_speech_resh = output_resh[usedbins_resh]
	    
	#apply kmeans clustering and assign each bin to a clustering
	kmeans_model=KMeans(n_clusters=self.nrS, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
	
	for _ in range(5):
	# Sometime it fails due to some indexerror and I'm not sure why. Just retry then. max 5 times
	    try:
		kmeans_model.fit(output_speech_resh)
	    except IndexError:
	      continue
	    break
	
	predicted_labels = kmeans_model.predict(output_resh)
	predicted_labels_resh = np.reshape(predicted_labels,[T,F])
	
	#reconstruct the masks from the cluster labels
	masks = np.zeros([self.nrS,T,F])
	for spk in range(self.nrS):
	    masks[spk,:,:] = predicted_labels_resh==spk
	
	return masks