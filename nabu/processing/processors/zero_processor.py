'''@file zero_processor.py
contains the ZeroProcessor class'''

import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory
import pdb

class ZeroProcessor(processor.Processor):
    '''a processor for audio files, this will return mask with all elements 0'''

    def __init__(self, conf, segment_lengths):
        '''ZeroProcessor constructor

        Args:
            conf: ScorelabelperfeatureProcessor configuration as a dict of strings
            segment_lengths: A list containing the desired lengths of segments.
            Possibly multiple segment lengths'''

        #create the feature computer
        self.comp = feature_computer_factory.factory(conf['feature'])(conf)

        #set the length of the segments. Possibly multiple segment lengths
        self.segment_lengths = segment_lengths

        #initialize the metadata
        self.nrS = int(conf['nrs'])

        self.dim = self.comp.get_dim()*self.nrS
        self.target_dim = self.comp.get_dim() * self.nrS
        self.nontime_dims=[self.target_dim]
        super(ZeroProcessor, self).__init__(conf)


    def __call__(self, dataline):
        '''process the data in dataline
        Args:
            dataline: either a path to a wav file or a command to read and pipe
                an audio file

        Returns:
            segmented_data: The segmented zeros
            utt_info: some info on the utterance'''

        utt_info= dict()

        rate, utt = _read_wav(dataline)


        features = self.comp(utt, rate)
        T = np.shape(features)[0]

        targets = np.zeros((T,self.dim))
        segmented_data = self.segment_data(targets)

        return segmented_data, utt_info


    def write_metadata(self, datadir):
        '''write the processor metadata to disk

        Args:
            dir: the directory where the metadata should be written'''

        for i,seg_length in enumerate(self.segment_lengths):
            seg_dir = os.path.join(datadir,seg_length)
            with open(os.path.join(seg_dir, 'dim'), 'w') as fid:
                fid.write(str(self.dim))
            with open(os.path.join(seg_dir, 'nontime_dims'), 'w') as fid:
                fid.write(str(self.nontime_dims)[1:-1])

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
