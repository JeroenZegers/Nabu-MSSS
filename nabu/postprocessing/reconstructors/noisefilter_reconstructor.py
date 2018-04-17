'''@file noisefilter_reconstructor.py
contains the reconstor class using noisefilter'''

import mask_reconstructor
import numpy as np
import os
import pdb

class NoiseFilterReconstructor(mask_reconstructor.MaskReconstructor):
    '''the noisefilter reconstructor class

    a reconstructor using deep clustering'''

    requested_output_names = ['noise_filter']

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''DeepclusteringReconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions will be stored
        '''

        super(NoiseFilterReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)


    def _get_masks(self, output, utt_info):
        '''estimate the masks

        Args:
            output: the output of a single utterance of the neural network
                utt_info: some info on the utterance

        Returns:
            the estimated masks'''

	    noise_filter = output['noise_filter']
	    #only the non-silence bins will be used for the clustering


    	[T,F] = np.shape(noise_filter)

    	masks = np.zeros([1,T,F])
        masks[0,:,:] = noise_filter

        return masks
