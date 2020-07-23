"""@file vad_timings_processor.py
contains the VadTimingsProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
import gzip
from nabu.processing.feature_computers import feature_computer_factory


class VadTimingsProcessor(processor.Processor):
	"""a processor for convert VAD timings into targets"""

	def __init__(self, conf, segment_lengths):
		"""VadTimingsProcessor constructor

		Args:
			conf: VadTimingsProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments. 
			Possibly multiple segment lengths"""

		# create the feature computer
		self.comp = feature_computer_factory.factory('frames')(conf)
		self.winlen = float(conf['winlen'])
		self.winstep = float(conf['winstep'])

		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths

		self.nrS = int(conf['nrs'])

		# initialize the metadata
		self.dim = self.nrS
		self.max_length = np.zeros(len(self.segment_lengths))
		# self.sequence_length_histogram = np.zeros(0, dtype=np.int32)
		self.nontime_dims = [self.dim]

		super(VadTimingsProcessor, self).__init__(conf)

	def __call__(self, datalines):
		"""process the data in dataline
		Args:
			datalines: in format 'mix_wav spk_id1 seg1_start seg1_end seg2_start seg2_end ... spk_id1 spk_id2 ... spk_id2

		Returns:
			segmented_data: The segmented features as a list of numpy arrays per segment length
			utt_info: some info on the utterance"""
			
		utt_info = dict()

		split_lines = datalines.split(' ')
		mix_file = split_lines.pop(0)

		# read the wav file
		rate, utt = _read_wav(mix_file)

		# compute the features
		frames = self.comp(utt, rate)

		audio_length = np.shape(frames)[-2]

		vad_indicator = np.zeros([audio_length, self.dim], dtype=np.bool)

		ind = 0
		spk_ind = 0
		new_id = True
		prev_id = ''
		while True:
			if new_id:
				prev_id = split_lines[ind]
				ind += 1
				new_id = False
			if prev_id == split_lines[ind]:
				ind += 1
				new_id = True
				spk_ind += 1
			else:
				seg_st = float(split_lines[ind])
				seg_st_frames = sec2frames(seg_st, self.winlen, self.winstep)
				if seg_st_frames > audio_length-1:
					seg_st_frames = audio_length-1
				seg_end = float(split_lines[ind+1])
				seg_end_frames = sec2frames(seg_end, self.winlen, self.winstep)
				if seg_end_frames > audio_length:
					seg_end_frames = audio_length

				vad_indicator[seg_st_frames:seg_end_frames, spk_ind] = 1

				ind += 2

			if ind >= len(split_lines):
				break
			
		# split the data for all desired segment lengths
		segmented_data = self.segment_data(vad_indicator)

		# update the metadata
		for i, seg_length in enumerate(self.segment_lengths):
			self.max_length[i] = max(self.max_length[i], np.shape(segmented_data[seg_length][0])[0])
			# seq_length = np.shape(segmented_data[seg_length][0])[0]
			# if seq_length >= np.shape(self.sequence_length_histogram[i])[0]:
			# self.sequence_length_histogram[i] = np.concatenate(
				# [self.sequence_length_histogram[i], np.zeros(
				# seq_length-np.shape(self.sequence_length_histogram[i])[0]+1,
				# dtype=np.int32)]
			# )
			# self.sequence_length_histogram[i][seq_length] += len(segmented_data[seg_length])

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


class VadTimings2SamplesProcessor(processor.Processor):
	"""a processor for convert VAD timings into time domain targets"""

	def __init__(self, conf, segment_lengths):
		"""VadTimingsProcessor constructor

		Args:
			conf: VadTimingsProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments.
			Possibly multiple segment lengths"""

		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths

		self.nrS = int(conf['nrs'])

		# initialize the metadata
		self.dim = self.nrS
		self.max_length = np.zeros(len(self.segment_lengths))
		# self.sequence_length_histogram = np.zeros(0, dtype=np.int32)
		self.nontime_dims = [self.dim]

		super(VadTimings2SamplesProcessor, self).__init__(conf)

	def __call__(self, datalines):
		"""process the data in dataline
		Args:
			datalines: in format 'mix_wav spk_id1 seg1_start seg1_end seg2_start seg2_end ... spk_id1 spk_id2 ... spk_id2

		Returns:
			segmented_data: The segmented features as a list of numpy arrays per segment length
			utt_info: some info on the utterance"""

		utt_info = dict()

		split_lines = datalines.split(' ')
		mix_file = split_lines.pop(0)

		# read the wav file
		rate, utt = _read_wav(mix_file)

		audio_length = len(utt)

		vad_indicator = np.zeros([audio_length, self.dim], dtype=np.bool)

		ind = 0
		spk_ind = 0
		new_id = True
		prev_id = ''
		while True:
			if new_id:
				prev_id = split_lines[ind]
				ind += 1
				new_id = False
			if prev_id == split_lines[ind]:
				ind += 1
				new_id = True
				spk_ind += 1
			else:
				seg_st = float(split_lines[ind])
				seg_st_samples = sec2samples(seg_st, rate)
				if seg_st_samples > audio_length - 1:
					seg_st_samples = audio_length - 1
				seg_end = float(split_lines[ind + 1])
				seg_end_samples = sec2samples(seg_end, rate)
				if seg_end_samples > audio_length:
					seg_end_samples = audio_length

				vad_indicator[seg_st_samples:seg_end_samples, spk_ind] = 1

				ind += 2

			if ind >= len(split_lines):
				break

		# split the data for all desired segment lengths
		segmented_data = self.segment_data(vad_indicator)

		# update the metadata
		for i, seg_length in enumerate(self.segment_lengths):
			self.max_length[i] = max(self.max_length[i], np.shape(segmented_data[seg_length][0])[0])
		# seq_length = np.shape(segmented_data[seg_length][0])[0]
		# if seq_length >= np.shape(self.sequence_length_histogram[i])[0]:
		# self.sequence_length_histogram[i] = np.concatenate(
		# [self.sequence_length_histogram[i], np.zeros(
		# seq_length-np.shape(self.sequence_length_histogram[i])[0]+1,
		# dtype=np.int32)]
		# )
		# self.sequence_length_histogram[i][seq_length] += len(segmented_data[seg_length])

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


def sec2frames(time_in_seconds, winlength, winstep):
	""" turn time in seconds into time in frames"""
	time_in_frames = (time_in_seconds - winlength/2)/winstep
	time_in_frames = int(round(time_in_frames))
	if time_in_frames < 0:
		time_in_frames = 0
	return time_in_frames


def sec2samples(time_in_seconds, rate):
	""" turn time in seconds into time in samples"""
	time_in_samples = int(round(time_in_seconds * rate))

	if time_in_samples < 0:
		time_in_samples = 0
	return time_in_samples


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
