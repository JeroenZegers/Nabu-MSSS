"""@file onehotperfeature_target_processor.py
contains the onehotperfeatureTargetProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory


class onehotperfeatureTargetProcessor(processor.Processor):
	"""a processor for audio files, this will compute the targets"""

	def __init__(self, conf, segment_lengths):
		"""onehotperfeatureTargetProcessor constructor

		Args:
			conf: onehotperfeatureTargetProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments.
			Possibly multiple segment lengths"""

		# create the feature computer
		self.comp = feature_computer_factory.factory(conf['feature'])(conf)

		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths

		# initialize the metadata
		self.nrS = int(conf['nrs'])
		self.dim = self.comp.get_dim() * self.nrS
		self.nontime_dims = [self.dim]

		super(onehotperfeatureTargetProcessor, self).__init__(conf)

	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file

		Returns:
			segmented_data: The segmented targets as a list of numpy arrays per segment length
			utt_info: some info on the utterance"""

		utt_info = dict()

		splitdatalines = dataline.strip().split(' ')

		clean_features = None
		for splitdataline in splitdatalines:
			# read the wav file
			rate, utt = _read_wav(splitdataline)

			# compute the features
			features = self.comp(utt, rate)
			features = np.expand_dims(features, 2)

			if clean_features is None:
				clean_features = features
			else:
				clean_features = np.append(clean_features, features, 2)

		winner = np.argmax(clean_features, axis=2)
		targets = np.empty([features.shape[0], self.dim], dtype=bool)
		for s_ind in range(self.nrS):
			targets[:, s_ind::self.nrS] = winner == s_ind

		# split the data for all desired segment lengths
		segmented_data = self.segment_data(targets)

		return segmented_data, utt_info

	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""

		for i, seg_length in enumerate(self.segment_lengths):
			seg_dir = os.path.join(datadir, seg_length)
			with open(os.path.join(seg_dir, 'nrS'), 'w') as fid:
				fid.write(str(self.nrS))
			with open(os.path.join(seg_dir, 'dim'), 'w') as fid:
				fid.write(str(self.dim))
			with open(os.path.join(seg_dir, 'nontime_dims'), 'w') as fid:
				fid.write(str(self.nontime_dims)[1:-1])


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
