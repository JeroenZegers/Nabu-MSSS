'''@file pit_l41_reconstructor.py
contains the reconstor class using PIT L41'''

from sklearn.cluster import KMeans
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import pdb

class PITL41Reconstructor(mask_reconstructor.MaskReconstructor):
    '''the PIT L41 reconstructor class

    a reconstructor using deep using PIT L41'''
    
    requested_output_names = ['bin_emb','spk_emb']

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''PITL41Reconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions will be stored
        '''
        
        super(PITL41Reconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)
                

    def _get_masks(self, output):
	'''estimate the masks

	Args:
	    output: the output of a single utterance of the neural network

	Returns:
	    the estimated masks'''

	bin_embeddings = output['bin_emb']
	spk_embeddings = output['spk_emb']
	
	spk_emb_dim = np.shape(spk_embeddings)[1]
	emb_dim = spk_emb_dim/self.nrS
	[T,output_dim] = np.shape(bin_embeddings)
	F = output_dim/emb_dim
	
	
	vi = np.reshape(bin_embeddings,[1,T,F,emb_dim])
	vi_norm = vi/np.linalg.norm(vi, axis=3, keepdims=True)
	vo = np.reshape(spk_embeddings,[self.nrS,1,1,emb_dim])
	vo_norm = vo/np.linalg.norm(vo, axis=3, keepdims=True)
	
	masks = 1.0/(np.linalg.norm(vo_norm-vi_norm, axis=3, keepdims=False))
	
	exp_masks = np.exp(masks)
	masks = exp_masks / np.sum(exp_masks,axis=0)
	
	return masks