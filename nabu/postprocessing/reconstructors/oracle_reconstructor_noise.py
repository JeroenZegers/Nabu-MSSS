'''@file reconstructor.py
contains the Reconstructor class'''

import os
import numpy as np
import pdb
import mask_reconstructor
from nabu.postprocessing import data_reader

from abc import ABCMeta, abstractmethod

class OracleReconstructor(mask_reconstructor.MaskReconstructor):
    '''the general reconstructor class

    a reconstructor is used to reconstruct the signals from the models output'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        super(OracleReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)

        #get the original mixtures reader
        noise_targets_name = conf['noise_targets']
        noise_targets_dataconf = dict(dataconf.items(noise_targets_name))
        self.noise_targets_reader = data_reader.DataReader(usedbins_dataconf,self.segment_lengths)
        binary_targets_name = conf['binary_targets']
        binary_targets_dataconf = dict(dataconf.items(noise_targets_name))
        self.binary_targets_reader = data_reader.DataReader()


    def _get_masks(self, output, utt_info):
        '''estimate the masks

        Args:
            output: the output of a single utterance of the neural network
            utt_info: some info on the utterance

        Returns:
            the estimated masks'''

        #only the non-silence bins will be used for the clustering
        noise_targets_complete, _ = self.noise_targets_reader(self.pos)
        noise_targets = noise_targets[:,self.nrS::self.nrS+1].astype(int)
        binary_targets,_ = self.binary_targets_reader(self.pos)

        #reconstruct the masks from the cluster labels
        masks = np.zeros([self.nrS,T,F])
        for spk in range(self.nrS):
            binary_targets_spk =  binary_targets[:,spk::,self.nrS].astype(int)
            masks[spk,:,:] = (1-noise_targets) *binary_targets_spk


        return masks
