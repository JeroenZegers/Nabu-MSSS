'''@file stackedmasks_reconstructor.py
contains the reconstor class using deep clustering'''

from sklearn.cluster import KMeans
import mask_reconstructor
import numpy as np

class StackedmasksReconstructor(mask_reconstructor.MaskReconstructor):
    '''the stacked masks reconstructor class

    a reconstructor using that uses stacked masks'''

    def __init__(self, conf, evalconf, dataconf, expdir, task):
        '''StackedmasksReconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configurationn
            expdir: the experiment directory
            task: name of the task
        '''
        
        super(StackedmasksReconstructor, self).__init__(conf, evalconf, dataconf, expdir, task)       

    def _get_masks(self, output):
	'''get the masks by simply destacking the stacked masks into separate masks

	Args:
	    output: the output of a single utterance of the neural network

	Returns:
	    the estimated masks'''
	    
	[T,target_dim] = np.shape(output)
	F = target_dim/self.nrS
	
	
	masks = output.reshape([T,F,self.nrS])
	masks = np.transpose(masks,[2,0,1])
	
	#apply softmax
	exp_masks = np.exp(masks)
	masks = exp_masks / np.sum(exp_masks,axis=0)
	
	return masks