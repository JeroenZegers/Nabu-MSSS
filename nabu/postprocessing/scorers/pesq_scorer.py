'''@file pesq_scorer.py
contains the scorer using PESQ'''

import scorer
import numpy as np
import os
import itertools
import subprocess
import pdb

class PESQScorer(scorer.Scorer):
    '''the PESQ scorer class. Uses build code from https://www.itu.int/rec/T-REC-P.862-200102-I/en
    Bad metric for multi speaker source separation?
    
    a scorer using PESQ'''
    	
    score_metrics = ('PESQ')
    score_scenarios = ('SS','base')
    score_expects = 'files'

    def __init__(self, conf, evalconf, dataconf, rec_dir, numbatches):
        '''PESQScorer constructor

        Args:
            conf: the scorer configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions are
            numbatches: the number of batches to process
        '''
        
        super(PESQScorer, self).__init__(conf, evalconf, dataconf, rec_dir, numbatches)


    def _get_score(self,org_src_filenames, base_filenames, rec_src_filenames):
        '''score the reconstructed utterances with respect to the original source signals

        Args:
            org_src_filenames: the original source signals, as a list of audio filenames
            base_filenames: the duplicated base signal (original mixture), as a list of audio filenames
            rec_src_filenames: the reconstructed source signals, as a list of audio filenames

        Returns:
            the score'''
        
	nrS=len(org_src_filenames)

	#compute all src - rec pairs and find optimal permutation
	pesq_allpairs = np.empty([nrS,nrS])
	for rec_ind in range(nrS):
	    for src_ind in range(nrS):
		pesq_allpairs[src_ind,rec_ind]=pesq(org_src_filenames[src_ind],
			    rec_src_filenames[rec_ind],8000)
	
	permutations = list(itertools.permutations(range(nrS),nrS))
	pesq_allpermutations=[]
	for perm in permutations:
	    tmp=[]
	    for spk_ind in range(nrS):
		tmp.append(pesq_allpairs[spk_ind,perm[spk_ind]])
	    pesq_allpermutations.append(np.mean(tmp))
	
	best_perm_ind = np.argmax(pesq_allpermutations)
	best_perm = permutations[best_perm_ind]
	
	score_dict=dict()
	for i,metric in enumerate(self.score_metrics):
	    score_dict[metric]=dict()
	    score_dict[metric][self.score_scenarios[0]]=[]
	    score_dict[metric][self.score_scenarios[1]]=[]
	    
	    for spk in range(nrS):
		score_dict[metric][self.score_scenarios[0]].append(pesq_allpairs[spk,best_perm[spk_ind]])
		score_dict[metric][self.score_scenarios[1]].append(pesq(org_src_filenames[spk],base_filenames[spk],8000))
	      
	  
	return score_dict     
      
def pesq(reference, degraded, sample_rate=None, program='pesq'):
    """ Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
    on reference and degraded speech samples comparison.
    Sample rate must be 8000 or 16000 (or can be defined reading reference file
    header).
    PESQ utility must be installed.
    """
    if not os.path.isfile(reference) or not os.path.isfile(degraded):
	raise ValueError('reference or degraded file does not exist')
    if not sample_rate:
	import wave
	w = wave.open(reference, 'r')
	sample_rate = w.getframerate()
	w.close()
    if sample_rate not in (8000, 16000):
	raise ValueError('sample rate must be 8000 or 16000')
    pdb.set_trace()
    args = ["nabu/postprocessing/scorers/pesq",'+%d'%int(sample_rate),reference,degraded]
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    last_line = out.split('\n')[-2]
    if not last_line.startswith('Prediction'):
	print out
	raise ValueError(last_line)
    return float(last_line.split()[-1])

