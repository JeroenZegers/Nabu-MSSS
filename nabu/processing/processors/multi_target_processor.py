"""@file multi_target_processor.py
contains the MultiTargetProcessor class"""


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from nabu.processing.feature_computers import feature_computer_factory
import gzip


class MultiTargetProcessor(processor.Processor):
	"""a processor for audio files, this will compute the multiple targets"""

	def __init__(self, conf, segment_lengths):
		"""MultiTargetProcessor constructor

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
		self.target_dim = self.comp.get_dim()
		self.nontime_dims = [self.target_dim, self.nrS]

		if 'mvn_type' in conf:
			self.mvn_type = conf['mvn_type']
		else:
			self.mvn_type = 'None'
		if self.mvn_type == 'global':
			self.obs_cnt = 0
			self.glob_mean = np.zeros([1, self.dim])
			self.glob_std = np.zeros([1, self.dim])
		elif self.mvn_type in ['local', 'none', 'None', 'from_files']:
			pass
		else:
			raise Exception('Unknown way to apply mvn: %s' % conf['mvn_type'])
		
		super(MultiTargetProcessor, self).__init__(conf)

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

		targets = None
		for splitdataline in splitdatalines:
			# read the wav file
			rate, utt = _read_wav(splitdataline)

			# compute the features
			features = self.comp(utt, rate)

			# mean and variance normalize the features
			if self.mvn_type == 'global':
				features = (features-self.glob_mean)/(self.glob_std+1e-12)
			elif self.mvn_type == 'local':
				features = (features-np.mean(features, 0))/(np.std(features, 0)+1e-12)

			features = np.expand_dims(features, 2)

			if targets is None:
				targets = features
			else:
				targets = np.append(targets, features, 2)
		
		# split the data for all desired segment lengths
		segmented_data = self.segment_data(targets)

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
							splitline = line.strip('\n').split(' ')
							utt_name = splitline[0]
							splitdatalines = splitline[1:]

							for splitdataline in splitdatalines:
								# process the dataline
								if loop_type == 'mean':
									self.acc_mean(splitdataline)
								elif loop_type == 'std':
									self.acc_std(splitdataline)

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
		acc_feat = np.sum(np.square(features - self.glob_mean), 0)

		# update the standard deviation
		self.glob_std += acc_feat

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
