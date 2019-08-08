"""@file spatial_feat_processor.py
contains the SpatialFeatProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
import gzip
from nabu.processing.feature_computers import feature_computer_factory


class SpatialFeatProcessor(processor.Processor):
	"""a processor for audio files, this will compute spatial features"""

	def __init__(self, conf, segment_lengths):
		"""AudioFeatProcessor constructor

		Args:
			conf: SpatialFeatProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments. 
			Possibly multiple segment lengths"""

		# create the feature computer
		self.comp = feature_computer_factory.factory(conf['feature'])(conf)
		# the channel pairs for which to compute the spatial features
		channel_pairs = conf['channels_pairs'].split(' ')
		channel_pairs = [map(int, pair.split('-')) for pair in channel_pairs]
		self.channel_pairs = [[ch - 1 for ch in pair] for pair in channel_pairs]  # python index starts at 0
		
		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths 

		# initialize the metadata
		self.dim = self.comp.get_dim() * 2 * len(channel_pairs)
		self.max_length = np.zeros(len(self.segment_lengths))
		# self.sequence_length_histogram = np.zeros(0, dtype=np.int32)
		self.nontime_dims = [self.dim]

		super(SpatialFeatProcessor, self).__init__(conf)

	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file

		Returns:
			segmented_data: The segmented features as a list of numpy arrays per segment length
			utt_info: some info on the utterance"""
			
		utt_info = dict()

		splitdatalines = dataline.strip().split(' ')

		spatial_feats = []
		for ch_pair in self.channel_pairs:
			datalines = [splitdatalines[ch] for ch in ch_pair]

			# read the wav file and compute the features
			rate1, utt1 = _read_wav(datalines[0])
			angs_spec_1 = self.comp(utt1, rate1)

			# read the wav file and compute the features
			rate2, utt2 = _read_wav(datalines[1])
			angs_spec_2 = self.comp(utt2, rate2)

			ang_diff = angs_spec_1 - angs_spec_2
			cos_ipd = np.cos(ang_diff)
			sin_ipd = np.sin(ang_diff)
			spatial_feat = np.concatenate([cos_ipd, sin_ipd], axis=1)

			spatial_feats.append(spatial_feat)

		spatial_feats = np.concatenate(spatial_feats, axis=1)

		# split the data for all desired segment lengths
		segmented_data = self.segment_data(spatial_feats)

		# update the metadata
		for i, seg_length in enumerate(self.segment_lengths):
			self.max_length[i] = max(self.max_length[i], np.shape(segmented_data[seg_length][0])[0])

		return segmented_data, utt_info

	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""

		for i, seg_length in enumerate(self.segment_lengths):
			seg_dir = os.path.join(datadir, seg_length)
			# with open(os.path.join(seg_dir, 'sequence_length_histogram.npy'), 'w') as fid:
			# np.save(fid, self.sequence_length_histogram[i])
			with open(os.path.join(seg_dir, 'max_length'), 'w') as fid:
				fid.write(str(self.max_length[i]))
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
