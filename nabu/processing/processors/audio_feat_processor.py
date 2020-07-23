"""@file audio_feat_processor.py
contains the AudioFeatProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
import gzip
from nabu.processing.feature_computers import feature_computer_factory


class AudioFeatProcessor(processor.Processor):
	"""a processor for audio files, this will compute features"""

	def __init__(self, conf, segment_lengths):
		"""AudioFeatProcessor constructor

		Args:
			conf: AudioFeatProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments. 
			Possibly multiple segment lengths"""

		# create the feature computer
		self.comp = feature_computer_factory.factory(conf['feature'])(conf)
		
		# set the length of the segments. Possibly multiple segment lengths
		self.segment_lengths = segment_lengths 

		# initialize the metadata
		self.dim = self.comp.get_dim()
		self.max_length = np.zeros(len(self.segment_lengths))
		# self.sequence_length_histogram = np.zeros(0, dtype=np.int32)
		self.nontime_dims = [self.dim]
		
		# set the type of mean and variance normalisation
		self.mvn_type = conf['mvn_type']
		if conf['mvn_type'] == 'global':
			self.obs_cnt = 0
			self.glob_mean = np.zeros([1, self.dim])
			self.glob_std = np.zeros([1, self.dim])
		elif conf['mvn_type'] in ['local', 'none', 'None', 'from_files']:
			pass
		else:
			raise Exception('Unknown way to apply mvn: %s' % conf['mvn_type'])

		super(AudioFeatProcessor, self).__init__(conf)

	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file

		Returns:
			segmented_data: The segmented features as a list of numpy arrays per segment length
			utt_info: some info on the utterance"""
			
		utt_info = dict()

		# read the wav file
		rate, utt = _read_wav(dataline)

		# compute the features
		features = self.comp(utt, rate)

		# mean and variance normalize the features
		if self.mvn_type == 'global':
			features = (features-self.glob_mean)/(self.glob_std+1e-12)
		elif self.mvn_type == 'local':
			features = (features-np.mean(features, 0))/(np.std(features, 0)+1e-12)
			
		# split the data for all desired segment lengths
		segmented_data = self.segment_data(features)

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

	def pre_loop(self, dataconf):
		"""before looping over all the data to process and store it, calculate the
		global mean and variance to normalize the features later on
		
		Args:
			dataconf: config file on the part of the database being processed"""
		if self.mvn_type == 'global':
			loop_types = ['mean', 'std']
		
			# calculate the mean and variance
			for loop_type in loop_types:
				# if the directory of mean and variance are pointing to the store directory,
				# this means that the mean and variance should be calculated here.
				if dataconf['meanandvar_dir'] == dataconf['store_dir']:	  
					for datafile in dataconf['datafiles'].split(' '):
						if datafile[-3:] == '.gz':
							open_fn = gzip.open
						else:
							open_fn = open
		
						# loop over the lines in the datafile
						for line in open_fn(datafile):
			
							# split the name and the data line
							splitline = line.strip().split(' ')
							utt_name = splitline[0]
							dataline = ' '.join(splitline[1:])
			
							# process the dataline
							if loop_type == 'mean':
								self.acc_mean(dataline)
							elif loop_type == 'std':
								self.acc_std(dataline)
			
					if loop_type == 'mean':
						self.glob_mean = self.glob_mean/float(self.obs_cnt)
						with open(os.path.join(dataconf['meanandvar_dir'], 'glob_mean.npy'), 'w') as fid:
							np.save(fid, self.glob_mean)
					elif loop_type == 'std':
						self.glob_std = np.sqrt(self.glob_std/float(self.obs_cnt))
						with open(os.path.join(dataconf['meanandvar_dir'], 'glob_std.npy'), 'w') as fid:
							np.save(fid, self.glob_std)
				else:
					# get mean and variance calculated on training set
					if loop_type == 'mean':
						with open(os.path.join(dataconf['meanandvar_dir'], 'glob_mean.npy')) as fid:
							self.glob_mean = np.load(fid)
					elif loop_type == 'std':
						with open(os.path.join(dataconf['meanandvar_dir'], 'glob_std.npy')) as fid:
							self.glob_std = np.load(fid)

	def acc_mean(self, dataline):
		"""accumulate the features to get the mean
		Args:
				dataline: either a path to a wav file or a command to read and pipe
					an audio file"""
		
		# read the wav file
		rate, utt = _read_wav(dataline)

		# compute the features
		features = self.comp(utt, rate)
		
		# accumulate the features
		acc_feat = np.sum(features, 0)
		
		# update the mean and observation count
		self.glob_mean += acc_feat 
		self.obs_cnt += features.shape[0]

	def acc_std(self, dataline):
		"""accumulate the features to get the standard deviation
		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file"""
		
		# read the wav file
		rate, utt = _read_wav(dataline)
		
		# compute the features
		features = self.comp(utt, rate)
		
		# accumulate the features
		acc_feat = np.sum(np.square(features-self.glob_mean), 0)
		
		# update the standard deviation
		self.glob_std += acc_feat 

	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""
			
		if self.mvn_type == 'global':
			with open(os.path.join(datadir, 'glob_mean.npy'), 'w') as fid:
				np.save(fid, self.glob_mean)
			with open(os.path.join(datadir, 'glob_std.npy'), 'w') as fid:
				np.save(fid, self.glob_std)
		
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
