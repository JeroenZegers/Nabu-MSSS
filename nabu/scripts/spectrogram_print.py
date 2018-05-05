import sys
import os
sys.path.append(os.getcwd())
from six.moves import configparser
from nabu.processing.processors import processor_factory
from nabu.processing.feature_computers import spec
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav

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

path = "/home/pieter/Documenten/KuLeuven/Thesis/Website/Fragmenten/DC_F_FPC/French_F_F_speaker2.wav"

spec_cfg = configparser.ConfigParser()
spec_cfg.read(os.path.join('./nabu/scripts', 'spectrogram_config.cfg'))
conf = dict(spec_cfg.items('processor'))

rate,signal = _read_wav(path)

spec_computer = spec.Spec(conf)
spectrogram = spec_computer.comp_feat(signal,rate)
print spectrogram
plt.specgram(signal, NFFT=256, Fs=rate)
plt.show()
