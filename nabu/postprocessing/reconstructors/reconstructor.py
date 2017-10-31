'''@file reconstructor.py
contains the Reconstructor class'''

from abc import ABCMeta, abstractmethod
import os
import scipy.io.wavfile as wav

class Reconstructor(object):
    '''the general reconstructor class

    a reconstructor is used to reconstruct the signals from the models output'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, evalconf, dataconf, expdir, task):
        '''Reconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            expdir: the experiment directory
            task: name of the task
        '''

        self.conf = conf
        self.dataconf = dataconf
        self.batch_size = int(evalconf.get('evaluator','batch_size'))
        self.segment_lengths = evalconf.get('evaluator','segment_length').split(' ')
        
        self.nrS = int(conf['nrs'])
        
        #create the directory to write down the reconstructions
        self.rec_dir = os.path.join(expdir,'reconstructions',task)
        if not os.path.isdir(self.rec_dir):
	    os.makedirs(self.rec_dir)
	for spk in range(self.nrS):
	    if not os.path.isdir(os.path.join(self.rec_dir,'s' + str(spk+1))):
		os.makedirs(os.path.join(self.rec_dir,'s' + str(spk+1)))
	    
        
        #the use of the position variable only works because in the evaluator the 
        #shuffle option in the data_queue is set to False!!
        self.pos = 0


    def __call__(self, batch_outputs, batch_sequence_lengths):
        ''' reconstruct the signals and write the audio files
        
        Args:
	    - batch_outputs: An array containing the batch outputs of the network
	    - batch_sequence_lengths: contains the sequence length for each utterance
        '''

	for utt_ind in range(self.batch_size):
	  
	    utt_output = batch_outputs[utt_ind][:batch_sequence_lengths[utt_ind],:]
	  
	    #reconstruct the singnals 
	    reconstructed_signals, utt_info = self.reconstruct_signals(utt_output)
	    
	    #make the audiofiles for the reconstructed signals
	    self.write_audiofile(reconstructed_signals, utt_info)
	    
	    self.pos += 1

    @abstractmethod
    def reconstruct_signals(self, output):
        '''reconstruct the signals

        Args:
            output: the output of a single utterance of the neural network

        Returns:
            the reconstructed signals'''

    
    def write_audiofile(self, reconstructed_signals, utt_info):
        '''write the audiofiles for the reconstructions

        Args:
            reconstructed_signals: the reconstructed signals for a single mixture
            utt_info: some info on the utterance
	'''
	
	for spk in range(self.nrS):
	    rec_dir = os.path.join(self.rec_dir,'s' + str(spk+1))
	    filename = os.path.join(rec_dir,utt_info['utt_name']+'.wav')
	    signal = reconstructed_signals[spk]
	    wav.write(filename, utt_info['rate'], signal)
	  