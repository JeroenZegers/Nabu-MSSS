"""@file postprocessor.py
contains the Postprocessor class"""

from abc import ABCMeta, abstractmethod
import os
import scipy.io.wavfile as wav
import numpy as np


class Postprocessor(object):
	"""the general postprocessor class

	a postprocessor is used to process reconstructed signals"""

	__metaclass__ = ABCMeta

	def __init__(self, conf, evalconf, expdir, rec_dir, postprocessors_name=None, name=None):
		"""Postprocessor constructor

		Args:
			conf: the postprocessor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			expdir: the experiment directory
			rec_dir: the directory where the reconstructions are
			postprocessors_name: name of the postprocessor task
		"""

		self.conf = conf
		self.segment_lengths = evalconf.get('evaluator', 'segment_length').split(' ')
		
		self.nrS = int(conf['nrs'])

		# create the directory to store the post processing data
		self.store_dir = os.path.join(expdir, name or type(self).__name__, postprocessors_name)
		if not os.path.isdir(self.store_dir):
			os.makedirs(self.store_dir)
		#for spk in range(self.nrS):
			#if not os.path.isdir(os.path.join(self.store_dir,'s' + str(spk+1))):
			#os.makedirs(os.path.join(self.store_dir,'s' + str(spk+1)))
		self.store_scp_file = os.path.join(self.store_dir, 'pointers.scp')

		self.rec_dir = rec_dir
		self.rec_scp_file = os.path.join(self.rec_dir, 'pointers.scp')

		num_utts_already_postprocessed = countlines_of_file(self.store_scp_file)
		self.start_ind = num_utts_already_postprocessed
		self.open_scp_files(from_start=self.start_ind == 0)
		self.rec_scp_fid = open(self.rec_scp_file, 'r')

		# the use of the position variable only works because in the evaluator the
		# shuffle option in the data_queue is set to False!!
		self.pos = 0

	def __call__(self):
		""" postprocess the utterances in the reconstruction dir
		
		"""
		for utt_ind, line in enumerate(self.rec_scp_fid):
			if utt_ind < self.start_ind:
				continue

			if np.mod(utt_ind, 10) == 0:
				print 'Postprocessing utterance %d' % utt_ind

			#get the reconstructed signals
			splitline = line.strip().split(' ')
			utt_name = splitline[0]
			rec_src_filenames = splitline[1:]
			#rec_src_signals = list()
			#for rec_src_filename in rec_src_filenames:
			#rate, utterance = wav.read(rec_src_filename)
			#rec_src_signals.append(utterance)

			#post process the reconstructed signals
			#postproc_data = self.postproc(rec_src_signals,rate)
			postproc_data = self.postproc(rec_src_filenames)

			#write the data to the store dir
			self.write_data(postproc_data, utt_name, rec_src_filenames)

		self.store_scp_fid.close()

	@abstractmethod
	def postproc(self, singals):
		"""postprocess the signals

		Args:
			output: the signals to be postprocessed

		Returns:
			the post processing data"""

	def write_data(self, postproc_data, utt_name, rec_src_filenames):
		"""write the postprocessed data of the reconstructions

		Args:
			postproc_data: a list of post proccesed data (1 signal per speaker)
			utt_name: the name of the utterance
			rec_src_filenames: the filename of the reconstructed audio signals
	"""

		filename = os.path.join(self.store_dir, utt_name)

		data_str = ''
		for ind, vector in enumerate(postproc_data):
			data_str += ' '.join(map(str, vector))
			if ind < len(postproc_data)-1:
				data_str += ', '

		with open(filename, 'w') as fid:
			fid.write(data_str)

		write_str = '%s %s %s \n' % (utt_name, rec_src_filenames[0], filename)
		self.store_scp_fid.write(write_str)

	def open_scp_files(self, from_start=True):
		if from_start:
			file_mode = 'w'
		else:
			file_mode = 'a+'
		self.store_scp_fid = open(self.store_scp_file, file_mode)


def countlines_of_file(file):
	if not os.path.isfile(file):
		num_lines = 0
	else:
		num_lines = sum(1 for line in open(file))

	return num_lines

