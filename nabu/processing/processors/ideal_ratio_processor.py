'''@file ideal_ratio_processor.py
contains the idealRatioProcessor class'''

import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory
import pdb

class IdealRatioProcessor(processor.Processor):
    '''a processor for audio files, this will compute the ideal ratio masks'''

    def __init__(self, conf, segment_lengths):
        '''IdealRatioProcessor constructor

        Args:
            conf: IdealRatioProcessor configuration as a dict of strings
            segment_lengths: A list containing the desired lengths of segments.
            Possibly multiple segment lengths'''

        #create the feature computer
        self.comp = feature_computer_factory.factory(conf['feature'])(conf)

        #set the length of the segments. Possibly multiple segment lengths
        self.segment_lengths = segment_lengths

        #initialize the metadata
        self.dim = self.comp.get_dim()
        self.nontime_dims=[self.dim]

        super(IdealRatioProcessor, self).__init__(conf)

    def __call__(self, dataline):
        '''process the data in dataline
        Args:
            dataline: either a path to a wav file or a command to read and pipe
                an audio file

        Returns:
            segmented_data: The segmented info on bins to be used for scoring as a list of numpy arrays per segment length
            utt_info: some info on the utterance'''

        utt_info= dict()

        splitdatalines = dataline.strip().split(' ')
        nrS = len(splitdatalines) - 1
        speaker_rate = None
        speaker_utt = None
        # Add speaker signals
        for s in range(nrS):
            #read the wav file
            rate, utt = _read_wav(splitdatalines[s])
            if speaker_rate is None:
                speaker_rate = rate
                speaker_utt = utt
            else:
                if speaker_rate != rate:
                    raise Exception('Unequal sampling rates!')
                if len(speaker_utt) != len(utt):
                    raise Exception('Unequal length')
                speaker_utt = speaker_utt + utt
        speaker_features = self.comp(speaker_utt, speaker_rate)
        speaker_features[speaker_features < 1e-30] = 0 # avoid problems for silent bins
        mix_rate,mix_utt = _read_wav(splitdatalines[-1])
        mixture_features = self.comp(mix_utt,mix_rate) + 1e-48 # avoid division by zero
        # calculate ideal ratio mask
        targets=speaker_features/mixture_features
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
