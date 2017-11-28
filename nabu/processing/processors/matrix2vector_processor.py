'''@file matrix2vector_processor.py
contains the Matrix2VectorProcessor class'''


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory
import json
import pdb

class Matrix2VectorProcessor(processor.Processor):
    '''a processor for converting matrices to vectors.'''

    def __init__(self, conf, segment_lengths):
        '''Matrix2VectorProcessor constructor

        Args:
            conf: Matrix2VectorProcessor configuration as a dict of strings
            segment_lengths: A list containing the desired lengths of segments. 
            Possibly multiple segment lengths'''
            
        #create the feature computer
        self.comp = feature_computer_factory.factory(conf['feature'])(conf)

        #the number of rows in the matrix
        self.nrCol = int(conf['nrcol'])
        self.nrS = int(conf['nrs'])
        self.dim = self.nrS*self.nrCol

        #set the length of the segments. Possibly multiple segment lengths
        self.segment_lengths = segment_lengths 
        
        #set the type of mean and variance normalisation
        self.mvn_type = conf['mvn_type']
        if conf['mvn_type'] == 'global':
	    self.obs_cnt = 0
	    self.glob_mean = np.zeros([self.dim])
	    self.glob_std = np.zeros([self.dim])
	elif conf['mvn_type'] == 'None':
	    pass
	else:
	    raise Exception('Unknown way to apply mvn: %s' % conf['mvn_type'])
        
        super(Matrix2VectorProcessor, self).__init__(conf)

    def __call__(self, dataline):
        '''process the data in dataline
        Args:
            dataline: contains the audio mixture, and the matrix file

        Returns:
            segmented_data: The segmented features as a list of a vector per segment length
            utt_info: some info on the utterance'''
        
        utt_info=dict()
        
        split_dataline = dataline.split(' ')
        audiofile = split_dataline[0]
        matrixfile = split_dataline[-1]
        
        matrix = open(matrixfile).read().strip().split(',')
        utt_info['nrS']=self.nrS
        vector=np.zeros(self.dim)
        for ind,matrix_row in enumerate(matrix):
	  vector[ind*self.nrCol:(ind+1)*self.nrCol]=map(float, matrix_row.split(' '))
	  
	#mean and variance normalize the features
        if self.mvn_type == 'global':
	    vector = (vector-self.glob_mean)/self.glob_std
	
	#get the number of frames from the mixture audiofile
        rate, utt = _read_wav(audiofile)
        features = self.comp(utt, rate)
        Nfram = np.shape(features)[0]
	            	    
	# split the data for all desired segment lengths
	segmented_data = self.segment_data(vector,Nfram)
	
        return segmented_data, utt_info
      
    def pre_loop(self, dataconf):
	'''before looping over all the data to process and store it, calculate the
	global mean and variance to normalize the features later on
	
	Args:
	    dataconf: config file on the part of the database being processed'''
	if self.mvn_type == 'global':
	    loop_types=['mean','std']
	    
	    #calculate the mean and variance
	    for loop_type in loop_types:
	  
		#if the directory of mean and variance are pointing to the store directory,
		#this means that the mean and variance should be calculated here.
		if dataconf['meanandvar_dir'] == dataconf['store_dir']:
		      
		    for datafile in dataconf['datafiles'].split(' '):
			if datafile[-3:] == '.gz':
			    open_fn = gzip.open
			else:
			    open_fn = open

			#loop over the lines in the datafile
			for line in open_fn(datafile):
			    #split the name and the data line
			    split_dataline = line.strip().split(' ')
			    matrixfile = split_dataline[-1]

			    matrix = open(matrixfile).read().strip().split(',')
			    vector=np.zeros(self.dim)
			    for ind,matrix_row in enumerate(matrix):
			      vector[ind*self.nrCol:(ind+1)*self.nrCol]=map(float, matrix_row.split(' '))
			      
			    #process the dataline
			    if loop_type == 'mean':
				self.glob_mean += vector
				self.obs_cnt += 1
			    elif loop_type == 'std':
				self.glob_std += np.square(vector-self.glob_mean)
			
		    if loop_type == 'mean':
			self.glob_mean = self.glob_mean/float(self.obs_cnt)
		    elif loop_type == 'std':
			self.glob_std = np.sqrt(self.glob_std/float(self.obs_cnt))
		else:
		    #get mean and variance calculated on training set
		    if loop_type == 'mean':
			with open(os.path.join(dataconf['meanandvar_dir'], 'glob_mean.npy')) as fid:
			  self.glob_mean = np.load(fid)
		    elif loop_type == 'std':
			with open(os.path.join(dataconf['meanandvar_dir'], 'glob_std.npy')) as fid:
			  self.glob_std = np.load(fid)
		      
    def write_metadata(self, datadir):
        '''write the processor metadata to disk

        Args:
            dir: the directory where the metadata should be written'''
		
	if self.mvn_type == 'global':
	    with open(os.path.join(datadir, 'glob_mean.npy'), 'w') as fid:
		np.save(fid, self.glob_mean)
	    with open(os.path.join(datadir, 'glob_std.npy'), 'w') as fid:
		np.save(fid, self.glob_std)
		
	for i,seg_length in enumerate(self.segment_lengths):
	    seg_dir = os.path.join(datadir,seg_length)
	    with open(os.path.join(seg_dir, 'dim'), 'w') as fid:
		fid.write(str(self.dim))
		
    def segment_data(self, data,N):
	'''Usually data is segmented by splitting an utterance into different parts
	(see processor.py). For this processor, we just replicate the vector
	multiple times.
	
	Args:
	    data: the data to be split 
	    N: the the number of frames. To seen how many segments are required
	    
	Returns:
	    the segmented data
	'''
	
	segmented_data = dict()
	
	for seg_length in self.segment_lengths:
	    if seg_length == 'full':
		seg_data = [data]
	    else:
		seg_len=int(seg_length)
		Nseg = int(np.floor(float(N)/float(seg_len)))
				
		if(Nseg) == 0:
		  seg_data = [data]
		  
		else:
		  
		  seg_data=[]
		  for seg_ind in range(Nseg):
		    seg_data.append(data)
	
	    
	    segmented_data[seg_length] = seg_data
	    
	return segmented_data

def _read_wav(wavfile):
    '''
    read a wav file

    Args:
        wavfile: either a path to a wav file or a command to read and pipe
            an audio file

    Returns:
        - the sampling rate
        - the utterance as a numpy array
    '''

    if os.path.exists(wavfile):
        #its a file
        (rate, utterance) = wav.read(wavfile)
    elif wavfile[-1] == '|':
        #its a command

        #read the audio file
        pid = subprocess.Popen(wavfile + ' tee', shell=True,
                               stdout=subprocess.PIPE)
        output, _ = pid.communicate()
        output_buffer = StringIO.StringIO(output)
        (rate, utterance) = wav.read(output_buffer)
    else:
        #its a segment of an utterance
        split = wavfile.split(' ')
        begin = float(split[-2])
        end = float(split[-1])
        unsegmented = ' '.join(split[:-2])
        rate, full_utterance = _read_wav(unsegmented)
        utterance = full_utterance[int(begin*rate):int(end*rate)]


    return rate, utterance
 