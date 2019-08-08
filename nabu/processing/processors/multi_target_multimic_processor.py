"""@file multi_target_multimic_processor.py
contains the MultiTargetMultimicProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory


class MultiTargetMultimicProcessor(processor.Processor):
	"""a processor for audio files, this will compute the multiple targets"""

	def __init__(self, conf, segment_lengths):
		"""MultiTargetMultimicProcessor constructor

		Args:
			conf: MultiTargetProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments.
			Possibly multiple segment lengths"""

		# create the feature computer
		self.comp = feature_computer_factory.factory(conf['feature'])(conf)

		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths
		self.dim = self.comp.get_dim()
		# initialize the metadata
		self.nrS = int(conf['nrs'])
		self.nr_channels = int(conf['nr_channels'])
		self.target_dim = self.comp.get_dim()
		self.nontime_dims = [self.target_dim, self.nrS]

		super(MultiTargetMultimicProcessor, self).__init__(conf)

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

		splitdatalines_per_spk = []
		for spk_ind in range(self.nrS):
			inds = range(spk_ind * self.nr_channels, (spk_ind+1) * self.nr_channels)
			splitdatalines_per_spk.append([splitdatalines[ind] for ind in inds])

		targets = None
		for splitdatalines in splitdatalines_per_spk:
			targets_spk = None
			for splitdataline in splitdatalines:
				# read the wav file
				rate, utt = _read_wav(splitdataline)

				# compute the features
				features = self.comp(utt, rate)
				features = np.expand_dims(features, 2)

				if targets_spk is None:
					targets_spk = features
				else:
					targets_spk = np.append(targets_spk, features, 2)

			targets_averaged = np.mean(targets_spk, 2, keepdims=True)

			if targets is None:
				targets = targets_averaged
			else:
				targets = np.append(targets, targets_averaged, 2)

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
				fid.write(str(self.target_dim))
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
		pid = subprocess.Popen(wavfile + ' tee', shell=True,  stdout=subprocess.PIPE)
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
