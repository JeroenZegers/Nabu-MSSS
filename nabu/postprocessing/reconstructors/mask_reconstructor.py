'''@file mask_reconstructor.py
contains the reconstructor class with use of a mask'''

import reconstructor
from nabu.postprocessing import data_reader
from nabu.processing.feature_computers import base
from abc import ABCMeta, abstractmethod

class MaskReconstructor(reconstructor.Reconstructor):
    '''the general reconstructor class using a mask

    a reconstructor using a mask'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''MaskReconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions will be stored
        '''
        
        super(MaskReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)
        
        #get the original mixtures reader 
        org_mix_name = conf['org_mix']
        org_mix_dataconf = dict(dataconf.items(org_mix_name))
        self.org_mix_reader = data_reader.DataReader(org_mix_dataconf, self.segment_lengths)


    def reconstruct_signals(self, output):
        '''reconstruct the signals

        Args:
            inputs: the output of a single utterance of the neural network

        Returns:
            the reconstructed signals
            some info on the utterance'''
            
        # get the masks    
        masks = self._get_masks(output)
        
        # get the original mixture
        mixture, utt_info= self.org_mix_reader(self.pos)
        
        # apply the masks to obtain the reconstructed signals. Use the conf for feature
        #settings from the original mixture
        reconstructed_signals = list()
        for spk in range(self.nrS):
	    spec_est = mixture * masks[spk,:,:]
	    reconstructed_signals.append(base.spec2time(spec_est, utt_info['rate'], 
						   utt_info['siglen'],
						   self.org_mix_reader.processor.comp.conf))
        
        return reconstructed_signals, utt_info
        
        
    @abstractmethod
    def _get_masks(self, output):
        '''estimate the masks

        Args:
            output: the output of a single utterance of the neural network

        Returns:
            the estimated masks'''
            
        

