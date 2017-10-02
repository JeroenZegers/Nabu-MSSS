'''@file sdr_scorer.py
contains the scorer using SDR'''

import scorer
import matlab.engine
import matlab
import numpy as np
import pdb

class SdrScorer(scorer.Scorer):
    '''the SDR scorer class

    a scorer using SDR'''
    	
    score_metrics = ('SDR','SIR','SAR')
    score_scenarios = ('SS','base')

    def __init__(self, conf, dataconf, rec_dir, numbatches):
        '''SdrScorer constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions are
            numbatches: the number of batches to process
        '''
        
        super(SdrScorer, self).__init__(conf, dataconf, rec_dir, numbatches)
        
        self.matlab_eng = matlab.engine.start_matlab("-nodesktop")
        #Go to the directory where the bss_eval_sources.m script is
	self.matlab_eng.cd('/users/spraak/jzegers/Documents/MATLAB/Toolboxes')


    def _get_score(self,org_src_signals, base_signals, rec_src_signals):
        '''score the reconstructed utterances with respect to the original source signals

        Args:
            org_src_signals: the original source signals, as a list of numpy arrarys
            base_signals: the duplicated base signal (original mixture), as a list of numpy arrarys
            rec_src_signals: the reconstructed source signals, as a list of numpy arrarys

        Returns:
            the score'''
        
	nrS=len(org_src_signals)
	
        #convert variables to matlab format
        matlab_mapping = lambda x:matlab.double(x)
        org_src_signals_mat = list_of_numpy2matlab(org_src_signals, matlab_mapping)
        base_signals_mat = list_of_numpy2matlab(base_signals, matlab_mapping)
        rec_src_signals_mat = list_of_numpy2matlab(rec_src_signals, matlab_mapping)
        
        #use the standard matlab script for scoring
        collect_outputs=dict()
        collect_outputs[self.score_scenarios[0]]= self.matlab_eng.bss_eval_sources(
							      rec_src_signals_mat,
							      org_src_signals_mat,
							      nargout=4)
        collect_outputs[self.score_scenarios[1]]= self.matlab_eng.bss_eval_sources(
							      base_signals_mat,
							      org_src_signals_mat,
							      nargout=4)
	
	#convert the matlab outputs to python format and put them in a single dictionary

        score_dict = dict()
	for i,metric in enumerate(self.score_metrics):
	    score_dict[metric]=dict()
	    
	    for j,scen in enumerate(self.score_scenarios):
		score_dict[metric][scen]=[]
		
		for spk in range(nrS):
		    score_dict[metric][scen].append(collect_outputs[scen][i][0][spk])
	  
	return score_dict     
     
    def finished_scoring(self):
	''' after scoring is done, close the MATLAB session'''
	
	self.matlab_eng.quit()
	

def list_of_numpy2matlab(var,mapping):
    '''map a list of numpy arrays, first to a regular list and finally map it to a 
    MATLAB array. The mapping does take some time however. Concidering skipping this and
    passing wavpath directly to MATLAB?
    
    Args:
	var: The variable that will be mapped
	mapping: a python lambda function, mapping a list to a MATLAB array of desired type
	
    Returns:
	a MATLAB array
    '''
  
    if type(var) is not list:
	raise TypeError('the input variable should be a list')
    
    var_list = list()
    for utt in var:
	if type(utt) is not np.ndarray:
	    raise TypeError('the input variable should be a list of numpy arrays')
	    
	uttlist = utt.tolist()
	var_list.append(uttlist)
	
    var_matlab = mapping(var_list)
    
    return var_matlab