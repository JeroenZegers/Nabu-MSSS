'''@file processor.py
contains the Processor class'''

from abc import ABCMeta, abstractmethod
import numpy as np

class Processor(object):
    '''general Processor class for data processing'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''Processor constructor

        Args:
            conf: processor configuration as a dictionary of strings
        '''

        self.conf = conf

    @abstractmethod
    def __call__(self, dataline):
        '''process the data in dataline
        Args:
            dataline: a string, can be a line of text a pointer to a file etc.

        Returns:
            The processed data'''
            
    def segment_data(self, data):
	'''split the data into segments for all desired segment lengths
	
	Args:
	    data: the data to be split in numpy format
	    
	Returns:
	    the segmented data
	'''
	
	
	segmented_data = dict()
	N = len(data)
	
	for seg_length in self.segment_lengths:
	    if seg_length == 'full':
		seg_data = [data]
	    else:
		seg_len=int(seg_length)
		Nseg = int(np.floor(float(N)/float(seg_len)))
		
		if(Nseg) == 0:
		  seg_data = np.concatenate((data,np.zeros(seg_len-N,self.dim)),axis=0)
		  
		else:
		  
		  seg_data=[]
		  for seg_ind in range(Nseg):
		    seg_data.append(data[seg_ind*seg_len:(seg_ind+1)*seg_len,:])
	
	    
	    segmented_data[seg_length] = seg_data
	    
	return segmented_data


    @abstractmethod
    def write_metadata(self, datadir):
        '''write the processor metadata to disk

        Args:
            dir: the directory where the metadata should be written'''
            
    def pre_loop(self, dataconf):
	'''allow the processor to access data before looping over all the data
	
	Args:
	    dataconf: config file on the part of the database being processed'''

    def post_loop(self, dataconf):
	'''allow the processor to access data after looping over all the data
	
	Args:
	    dataconf: config file on the part of the database being processed'''