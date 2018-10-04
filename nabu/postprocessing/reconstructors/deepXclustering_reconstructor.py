'''@file deepXclustering_reconstructor.py
contains the reconstor class using deep clustering, using Xmeans'''

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import pdb

class DeepXclusteringReconstructor(mask_reconstructor.MaskReconstructor):
    '''the deepxclustering reconstructor class

    a reconstructor using deep clustering, using Xmeans'''

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''DeepclusteringXReconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions will be stored
        '''
        
        super(DeepXclusteringReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)
        
        #get the usedbins reader
        usedbins_name = conf['usedbins']
        usedbins_dataconf = dict(dataconf.items(usedbins_name))
        self.usedbins_reader = data_reader.DataReader(usedbins_dataconf,self.segment_lengths)
        

    def _get_masks(self, output, utt_info):
	'''estimate the masks

	Args:
	    output: the output of a single utterance of the neural network
            utt_info: some info on the utterance

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
 
	#apply Xmeans clustering and assign each bin to a clustering
	k = 1
	BIC=1e20
	AIC=1e20
	while True:
	    kmeans_model_new=KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
	    
	    for _ in range(5):
	    # Sometime it fails due to some indexerror and I'm not sure why. Just retry then. max 5 times
		try:
		    kmeans_model_new.fit(output_speech_resh)
		except IndexError:
		  continue
		break
	    
	    RSS = kmeans_model_new.inertia_
	    BIC_new = np.log(np.shape(output_speech_resh)[0])*emb_dim*k + RSS
	    AIC_new = 2*emb_dim * k + RSS
	    
	    if BIC_new < BIC or AIC_new < AIC:
		k+= 1
		BIC=BIC_new
		kmeans_model = kmeans_model_new
		continue
	    else:
		if BIC_new > BIC:
		    print 'BIC increased, stopping search'
		else:
		    print 'AIC increased, stopping search'
		print '%d is found to be the optimal number of clusters' %k
		break

	#RSS=[]
	#BIC=[]
	#AIC=[]

	#for k in range(1,10):
	    #kmeans_model_new=KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
	    #labels=kmeans_model_new.fit_predict(output_speech_resh)
	    #RSS_new = kmeans_model_new.inertia_
	    #BIC_new = np.log(np.shape(output_speech_resh)[0])*emb_dim*k + RSS_new
	    #AIC_new = 2*emb_dim * k + RSS_new
	    #RSS.append(RSS_new)
	    #BIC.append(BIC_new)
	    #AIC.append(AIC_new)
	    	
	if k==self.nrS:
	    print 'found the correct number of clusters (%d)'%k
	else:
	    print 'found the wrong number of clusters (%d instead of %d)'%(k,self.nrS)
	    raise 'not yet implemented what to do next'
	predicted_labels = kmeans_model.predict(output_resh)
	predicted_labels_resh = np.reshape(predicted_labels,[T,F])
	
	#reconstruct the masks from the cluster labels
	masks = np.zeros([self.nrS,T,F])
	for spk in range(self.nrS):
	    masks[spk,:,:] = predicted_labels_resh==spk
	
	return masks