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
                    tensor of dimension [Txfeature_dimension*emb_dim]

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
        output_speech_resh = output_resh[usedbins_resh] # dim:N x embdim
	
        #apply kmeans clustering and assign each bin to a clustering
        kmeans_model=KMeans(n_clusters=self.nrS, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
        kmeans_model.fit(output_speech_resh)
        
        A = kmeans_model.cluster_centers_ # dim: nrS x embdim
        
        
        prod_1 = tf.matmul(A,output_resh,transpose_a=False, transpose_b = True,name='AVT')
		ones_M = tf.ones([nb_sources,N],name='ones_M')
		M = tf.divide(ones_M,ones_M+tf.exp(prod_1)) # dim: number_sources x N
	    
        #reconstruct the masks from the cluster labels
        mask = np.reshape(M,[self.nrS,T,F])
	
        return masks
