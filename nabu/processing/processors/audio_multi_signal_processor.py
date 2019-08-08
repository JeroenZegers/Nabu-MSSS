"""@file audio_multi_signal_processor.py
contains the AudioMultiSignalProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory


class AudioMultiSignalProcessor(processor.Processor):
	"""a processor for multiple audio signals"""

	def __init__(self, conf, segment_lengths):
		"""AudioMultiSignalProcessor constructor

		Args:
			conf: AudioMultiSignalProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments.
			Possibly multiple segment lengths"""

		# create the feature computer
		self.comp = feature_computer_factory.factory(conf['feature'])(conf)

		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths

		# initialize the metadata
		self.dim = self.comp.get_dim()

		super(AudioMultiSignalProcessor, self).__init__(conf)

	def __call__(self, dataline):
		"""process the data in dataline.
		Warning: will not work in combination with a tfwriter since the output
		is a list and not numpy array

		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file

		Returns:
			segmented_data: The multiple audio signals as list of numpy arrays per signal per segment length
			utt_info: some info on the utterance"""

		utt_info= dict()

		splitdatalines = dataline.strip().split(' ')

		multi_signal = list()
		for splitdataline in splitdatalines:
			# read the wav file
			rate, utt = _read_wav(splitdataline)

			# compute the features
			features = self.comp(utt, rate)

			multi_signal.append(features)

		multi_signal = np.array(multi_signal)
		# split the data for all desired segment lengths
		winlen_sample = int(rate*float(self.conf['winlen']))
		winstep_sample = int(rate*float(self.conf['winstep']))
		segmented_data = self.segment_data(
			multi_signal, time_dim=1, winlen_sample=winlen_sample, winstep_sample=winstep_sample)

		utt_info['rate'] = rate
		utt_info['nrSig'] = len(splitdatalines)
	
		return segmented_data, utt_info

	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""

		for i, seg_length in enumerate(self.segment_lengths):
			seg_dir = os.path.join(datadir, seg_length)

			with open(os.path.join(seg_dir, 'dim'), 'w') as fid:
				fid.write(str(self.dim))


def _read_wav(wavfile):
	"""
	read a wav file

	Args:
		wavfile: either a path to a wav file or a command to read and pipe
			an audio file

	Returns:
		- the sampling rate
		- the utterance as a numpy array
	"""

	if os.path.exists(wavfile):
		# its a file
		(rate, utterance) = wav.read(wavfile)
	elif wavfile[-1] == '|':
		# its a command

		# read the audio file
		pid = subprocess.Popen(wavfile + ' tee', shell=True, stdout=subprocess.PIPE)
		output, _ = pid.communicate()
		output_buffer = StringIO.StringIO(output)
		(rate, utterance) = wav.read(output_buffer)
	else:
		# its a segment of an utterance
		split = wavfile.split(' ')
		begin = float(split[-2])
		end = float(split[-1])
		unsegmented = ' '.join(split[:-2])
		rate, full_utterance = _read_wav(unsegmented)
		utterance = full_utterance[int(begin*rate):int(end*rate)]

	return rate, utterance

