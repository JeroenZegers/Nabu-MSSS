'''@file stackedmasks_reconstructor.py
contains the reconstor class using deep clustering'''

from sklearn.cluster import KMeans
import mask_reconstructor
import numpy as np

class StackedmasksReconstructor(mask_reconstructor.MaskReconstructor):
    '''the stacked masks reconstructor class

    a reconstructor using that uses stacked masks'''
    
    requested_output_names = ['bin_est']

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''StackedmasksReconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions will be stored
        '''
        
        super(StackedmasksReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)       

    def _get_masks(self, output,utt_info):
	'''get the masks by simply destacking the stacked masks into separate masks and
	normalizing them with softmax

	Args:
	    output: the output of a single utterance of the neural network
            utt_info: some info on the utterance

	Returns:
	    the estimated masks'''
	    
	[T,target_dim] = np.shape(output['bin_est'])
	F = target_dim/self.nrS
	
	
	masks = output['bin_est'].reshape([T,F,self.nrS])
	masks = np.transpose(masks,[2,0,1])
	
	#apply softmax
	exp_masks = np.exp(masks)
	masks = exp_masks / np.sum(exp_masks,axis=0)
	
	return masks