"""@file parallel_reconstructor.py
contains the ParallelReconstructor class"""

from abc import ABCMeta, abstractmethod
import os
import scipy.io.wavfile as wav
import numpy as np
from joblib import Parallel, delayed


class ParallelReconstructor(object):
	"""the general reconstructor class

	a reconstructor is used to reconstruct the signals from the models output"""

	__metaclass__ = ABCMeta

	def __init__(self, conf, evalconf, dataconf, rec_dir, task):
		"""ParallelReconstructor constructor

		Args:
			conf: the reconstructor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions will be stored
		"""

		self.conf = conf
		self.dataconf = dataconf
		if evalconf.has_option(task, 'batch_size'):
			self.batch_size = int(evalconf.get(task, 'batch_size'))
		else:
			self.batch_size = int(evalconf.get('evaluator', 'batch_size'))
		self.segment_lengths = evalconf.get('evaluator', 'segment_length').split(' ')
		
		self.nrS = int(conf['nrs'])
		
		# create the directory to write down the reconstructions
		self.rec_dir = rec_dir
		if not os.path.isdir(self.rec_dir):
			os.makedirs(self.rec_dir)
		for spk in range(self.nrS):
			if not os.path.isdir(os.path.join(self.rec_dir, 's' + str(spk+1))):
				os.makedirs(os.path.join(self.rec_dir, 's' + str(spk+1)))

		# the use of the position variable only works because in the evaluator the
		# shuffle option in the data_queue is set to False!!
		self.pos = 0

		self.scp_file = open(os.path.join(self.rec_dir, 'pointers.scp'), 'w')
		
		# Wheter the raw output should also be stored (besides the reconstructed audiosignal)
		self.store_output = conf['store_output'] == 'True'
		if self.store_output:
			self.output_dir = os.path.join(rec_dir, 'raw_output')
			if not os.path.isdir(self.output_dir):
				os.makedirs(self.output_dir)

	def __call__(self, batch_outputs, batch_sequence_lengths):
		""" reconstruct the signals and write the audio files
		
		Args:
		- batch_outputs: A dictionary containing the batch outputs of the network
		- batch_sequence_lengths: A dictionary containing the sequence length for each utterance
		"""

		Parallel(n_jobs=6)(delayed(self.reconstruct_1_signal)(
			i, batch_outputs, batch_sequence_lengths) for i in range(self.batch_size))

		self.pos += self.batch_size

	def reconstruct_1_signal(self, utt_ind, batch_outputs, batch_sequence_lengths):
		utt_output = dict()
		for output_name in self.requested_output_names:
			utt_output[output_name] = \
				batch_outputs[output_name][utt_ind][:batch_sequence_lengths[output_name][utt_ind], :]

		# reconstruct the singnals
		reconstructed_signals, utt_info = self.reconstruct_signals(self.pos+utt_ind, utt_output)

		# make the audiofiles for the reconstructed signals
		self.write_audiofile(reconstructed_signals, utt_info)

		# if requested store the raw output
		if self.store_output:
			for output_name in self.requested_output_names:
				savename = output_name+'_'+utt_info['utt_name']
				np.save(os.path.join(self.output_dir, savename), utt_output[output_name])

	@abstractmethod
	def reconstruct_signals(self, output):
		"""reconstruct the signals

		Args:
			output: the output of a single utterance of the neural network

		Returns:
			the reconstructed signals"""

	def write_audiofile(self, reconstructed_signals, utt_info):
		"""write the audiofiles for the reconstructions

		Args:
			reconstructed_signals: the reconstructed signals for a single mixture
			utt_info: some info on the utterance
		"""

		write_str=utt_info['utt_name']
		for spk in range(self.nrS):
			rec_dir = os.path.join(self.rec_dir, 's' + str(spk+1))
			filename = os.path.join(rec_dir, utt_info['utt_name']+'.wav')
			signal = reconstructed_signals[spk]
			wav.write(filename, utt_info['rate'], signal)
			write_str += ' ' + filename

		write_str += ' \n'
		self.scp_file.write(write_str)
