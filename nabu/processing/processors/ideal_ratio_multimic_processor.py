"""@file ideal_ratio_multimic_processor.py
contains the IdealRatioMultimicProcessor class"""

import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory


class IdealRatioMultimicProcessor(processor.Processor):
	"""a processor for audio files, this will compute the ideal ratio masks. Actually, it is wiener filter, because
	featcomputer is expected to be energy"""

	def __init__(self, conf, segment_lengths):
		"""IdealRatioProcessor constructor

		Args:
			conf: IdealRatioProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments.
			Possibly multiple segment lengths"""

		# create the feature computer
		if 'pow' not in conf['feature']:
			raise Exception('expecting feature to be in power domain')
		self.comp = feature_computer_factory.factory(conf['feature'])(conf)

		if 'apply_sqrt' in conf and conf['apply_sqrt'] != 'True':
			self.apply_sqrt = False
		else:
			self.apply_sqrt = True

		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths

		# initialize the metadata
		self.nr_channels = int(conf['nr_channels'])
		self.dim = self.comp.get_dim()
		self.nontime_dims = [self.dim]

		super(IdealRatioMultimicProcessor, self).__init__(conf)

	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file

		Returns:
			segmented_data: The segmented info on bins to be used for scoring as a list of numpy arrays per segment length
			utt_info: some info on the utterance"""

		utt_info = dict()

		splitdatalines = dataline.strip().split(' ')

		splitdatalines_per_src = []
		for src_ind in range(len(splitdatalines)/self.nr_channels):
			inds = range(src_ind * self.nr_channels, (src_ind+1) * self.nr_channels)
			splitdatalines_per_src.append([splitdatalines[ind] for ind in inds])

		nr_spk = len(splitdatalines_per_src) - 1

		speaker_features = None
		for spk_ind in range(nr_spk):
			splitdatalines = splitdatalines_per_src[spk_ind]
			src_features = None
			for splitdataline in splitdatalines:
				# read the wav file
				rate, utt = _read_wav(splitdataline)

				# compute the features
				features = self.comp(utt, rate)
				features = np.expand_dims(features, 2)

				if src_features is None:
					src_features = features
				else:
					src_features = np.append(src_features, features, 2)

			src_features_averaged = np.mean(src_features, 2, keepdims=False)

			if speaker_features is None:
				speaker_features = src_features_averaged
			else:
				speaker_features += src_features_averaged

		ref_features = None
		for splitdataline in splitdatalines_per_src[-1]:
			# read the wav file
			rate, utt = _read_wav(splitdataline)

			# compute the features
			features = self.comp(utt, rate)
			features = np.expand_dims(features, 2)

			if ref_features is None:
				ref_features = features
			else:
				ref_features = np.append(ref_features, features, 2)

		ref_features_averaged = np.mean(ref_features, 2, keepdims=False)

		# calculate ideal ratio mask
		targets = speaker_features/(speaker_features + ref_features_averaged + 1e-48)
		if self.apply_sqrt:
			targets = np.sqrt(targets)
		segmented_data = self.segment_data(targets)

		return segmented_data, utt_info

	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""

		for i, seg_length in enumerate(self.segment_lengths):
			seg_dir = os.path.join(datadir, seg_length)
			with open(os.path.join(seg_dir, 'dim'), 'w') as fid:
				fid.write(str(self.dim))
			with open(os.path.join(seg_dir, 'nontime_dims'), 'w') as fid:
				fid.write(str(self.nontime_dims)[1:-1])


class SnrMultimicProcessor(IdealRatioMultimicProcessor):
	"""a processor for audio files, this will compute the snr"""
	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file

		Returns:
			segmented_data: The segmented info on bins to be used for scoring as a list of numpy arrays per segment length
			utt_info: some info on the utterance"""

		utt_info = dict()

		splitdatalines = dataline.strip().split(' ')

		splitdatalines_per_src = []
		for src_ind in range(len(splitdatalines)/self.nr_channels):
			inds = range(src_ind * self.nr_channels, (src_ind+1) * self.nr_channels)
			splitdatalines_per_src.append([splitdatalines[ind] for ind in inds])

		nr_spk = len(splitdatalines_per_src) - 1

		speaker_features = None
		for spk_ind in range(nr_spk):
			splitdatalines = splitdatalines_per_src[spk_ind]
			src_features = None
			for splitdataline in splitdatalines:
				# read the wav file
				rate, utt = _read_wav(splitdataline)

				# compute the features
				features = self.comp(utt, rate)
				features = np.expand_dims(features, 2)

				if src_features is None:
					src_features = features
				else:
					src_features = np.append(src_features, features, 2)

			src_features_averaged = np.mean(src_features, 2, keepdims=False)

			if speaker_features is None:
				speaker_features = src_features_averaged
			else:
				speaker_features += src_features_averaged

		ref_features = None
		for splitdataline in splitdatalines_per_src[-1]:
			# read the wav file
			rate, utt = _read_wav(splitdataline)

			# compute the features
			features = self.comp(utt, rate)
			features = np.expand_dims(features, 2)

			if ref_features is None:
				ref_features = features
			else:
				ref_features = np.append(ref_features, features, 2)

		ref_features_averaged = np.mean(ref_features, 2, keepdims=False)

		# calculate the SNR
		snr = 10*np.log10(speaker_features/(ref_features_averaged + 1e-48))

		segmented_data = self.segment_data(snr)

		return segmented_data, utt_info


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
