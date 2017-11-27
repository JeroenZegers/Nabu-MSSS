'''@file ivector_extractor.py
contains the IvectorExtractor class'''
import os
import numpy as np
import postprocessor
from nabu.processing.feature_computers import feature_computer_factory
import pdb

class IvectorExtractor(postprocessor.Postprocessor):
    '''the ivector extractor class

    a ivector extractor is used to extract ivectors from (reconstructed) signals'''

    def __init__(self, conf,proc_conf, evalconf,  expdir, rec_dir, task, name=None):
        '''IvectorExtractor constructor

        Args:
            conf: the ivector_extractor configuration as a dictionary
            proc_conf: the processor configuration for the postprocessor, as a dict
            evalconf: the evaluator configuration as a ConfigParser
            expdir: the experiment directory
            rec_dir: the directory where the reconstructions are
            task: name of the task
        '''
        
        super(IvectorExtractor, self).__init__(conf,proc_conf, evalconf, expdir, rec_dir, task, name)

	#The directory where all models for iVector extraction are stored. E.g. the 
	#Universal Background Model (UBM), the total variability matrix, ... 
        self.model_dir = conf['model_dir']
        
        UBM_w_file = os.path.join(self.model_dir,'UBM_w.npy')
        self.UBM_w = np.load(UBM_w_file)
        UBM_mu_file = os.path.join(self.model_dir,'UBM_mu.npy')
        self.UBM_mu = np.load(UBM_mu_file)
        UBM_sigma_file = os.path.join(self.model_dir,'UBM_sigma.npy')
        self.UBM_sigma = np.load(UBM_sigma_file)
        
	TV_file = os.path.join(self.model_dir,'T%d.npy'%(int(conf['tv_dim'])))
        self.TV = np.load(TV_file)
        
        self.apply_lda = conf['lda'] == 'True'
        if self.apply_lda:
	    V_file = os.path.join(self.model_dir,'V.npy')
	    self.V = np.load(V_file)
	    self.V = self.V[:int(conf['v_dim']),:]
  
	#create the feature computer
        self.comp = feature_computer_factory.factory(proc_conf['feature'])(proc_conf)
        self.VAD_thres = float(proc_conf['vad_thres'])
	    

    def postproc(self, signals, rate):
        '''postprocess the signals

        Args:
            output: the signals to be postprocessed

        Returns:
            the post processing data'''
        pdb.set_trace()    
        for signal in signals:    
            
	    mfcc = self.comp(signal, rate)
	    
	    logEne = mfcc[:,-1] 
	    mfcc = mfcc[:,:-1]
	    
	    logEne_thr = np.max(logEne)-np.log(self.VAD_thres)
	    VAD = logEne > logEne_thr
	    
	    mfcc = mfcc[VAD,:]
	    
	    #Problem that method for calculating mfcc differs from the method implemented in MATLAB that
	    #was used to determine the total variability space.
            
        
        
        return data