"""@file index_processor.py
contains the indexProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory
import json


class indexProcessor(processor.Processor):
	"""a processor for that simply read integers."""

	def __init__(self, conf, segment_lengths):
		"""indexProcessor constructor

		Args:
			conf: indexProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments. 
			Possibly multiple segment lengths"""
			
		# create the feature computer
		self.comp = feature_computer_factory.factory(conf['feature'])(conf)

		# create the string label to index dictionary
		self.nrS = int(conf['nrs'])

		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths 
		
		super(indexProcessor, self).__init__(conf)

	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: contains the audio mixture and the indices

		Returns:
			segmented_data: The segmented features as a list of indices per segment length
			utt_info: some info on the utterance"""
		
		utt_info = dict()
		
		split_dataline = dataline.split(' ')
		audiofile = split_dataline[0]
		string_labels = split_dataline[1:]
		
		utt_info['nrS'] = self.nrS
		index_labels = map(int, string_labels)
		# convert to python indexing
		index_labels = [lab-1 for lab in index_labels]
	
		# get the number of frames from the mixture audio file
		rate, utt = _read_wav(audiofile)
		features = self.comp(utt, rate)
		Nfram = np.shape(features)[0]
						
		# split the data for all desired segment lengths
		segmented_data = self.segment_data(index_labels, Nfram)
	
		return segmented_data, utt_info
	
	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""

		for i, seg_length in enumerate(self.segment_lengths):
			seg_dir = os.path.join(datadir, seg_length)

			with open(os.path.join(seg_dir, 'nrS'), 'w') as fid:
				fid.write(str(self.nrS))

	def segment_data(self, data, N):
		"""Usually data is segmented by splitting an utterance into different parts
		(see processor.py). For this processor, we just replicate the label index
		multiple times.

		Args:
			data: the data to be split
			N: the the number of frames. To seen how many segments are required

		Returns:
			the segmented data
		"""

		segmented_data = dict()

		for seg_length in self.segment_lengths:
			if seg_length == 'full':
				seg_data = [data]
			else:
				seg_len = int(seg_length)
				Nseg = int(np.floor(float(N)/float(seg_len)))

				if Nseg == 0:
					seg_data = [data]
				else:
					seg_data = []
					for seg_ind in range(Nseg):
						seg_data.append(data)

			segmented_data[seg_length] = seg_data

		return segmented_data


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
