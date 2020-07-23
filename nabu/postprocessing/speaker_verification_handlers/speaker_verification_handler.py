"""@file speaker_verification_handler.py
contains the SpeakerVerificationHandler class"""

from abc import ABCMeta, abstractmethod
import os
import numpy as np
from nabu.postprocessing import data_reader


class SpeakerVerificationHandler(object):
	"""the general speaker verification handler class

	a speaker verification handler is used to handle the models output for speaker verification purposes"""

	__metaclass__ = ABCMeta

	def __init__(self, conf, evalconf, dataconf, store_dir, exp_dir, task):
		"""Reconstructor constructor

		Args:
			conf: the reconstructor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			store_dir: the directory where the handled data will be stored
		"""

		self.conf = conf
		self.dataconf = dataconf
		if evalconf.has_option(task, 'batch_size'):
			self.batch_size = int(evalconf.get(task, 'batch_size'))
		else:
			self.batch_size = int(evalconf.get('evaluator', 'batch_size'))
		self.segment_lengths = evalconf.get('evaluator', 'segment_length').split(' ')

		self.nrS = int(conf['nrs'])

		if 'transpose_order' in conf:
			self.transpose_order = map(int, conf['transpose_order'].split(' '))
		else:
			self.transpose_order = False

		if 'cut_to_seq_length' not in conf or conf['cut_to_seq_length'] == 'True':
			self.cut_to_seq_length = True
		else:
			self.cut_to_seq_length = False

		# create the directory to write down the reconstructions
		self.store_dir = store_dir
		if not os.path.isdir(self.store_dir):
			os.makedirs(self.store_dir)
		if not os.path.isdir(os.path.join(self.store_dir, 'data')):
			os.makedirs(os.path.join(self.store_dir, 'data'))
		# for spk in range(self.nrS):
		# 	if not os.path.isdir(os.path.join(self.store_dir, 's' + str(spk+1))):
		# 		os.makedirs(os.path.join(self.store_dir, 's' + str(spk+1)))

		# the use of the position variable only works because in the evaluator the
		# shuffle option in the data_queue is set to False!!
		self.pos = 0

		# Whether the raw output should also be stored
		self.store_output = conf['store_output'] == 'True'
		if self.store_output:
			self.raw_output_dir = os.path.join(store_dir, 'raw_output')
			if not os.path.isdir(self.raw_output_dir):
				os.makedirs(self.raw_output_dir)

		# get the feature input reader, only to get the name of the utterance actually.
		input_features_names = conf['input_features'].split(' ')
		input_features_dataconfs = []
		for input_features_name in input_features_names:
			input_features_dataconfs.append(dict(dataconf.items(input_features_name)))
		self.input_features_reader = data_reader.DataReader(input_features_dataconfs, self.segment_lengths)

	def __call__(self, batch_outputs, batch_sequence_lengths):
		""" handle the output and store it

		Args:
		- batch_outputs: A dictionary containing the batch outputs of the network
		- batch_sequence_lengths: A dictionary containing the sequence length for each utterance
		"""
		if self.transpose_order:
			for output_name in self.requested_output_names:
				batch_outputs[output_name] = np.transpose(batch_outputs[output_name], self.transpose_order)

		for utt_ind in range(self.batch_size):

			utt_name = self.input_features_reader.get_name_for_pos(self.pos)

			utt_output = dict()
			for output_name in self.requested_output_names:
				utt_output[output_name] = batch_outputs[output_name][utt_ind]
				if self.cut_to_seq_length:
					utt_output[output_name] = \
						utt_output[output_name][:batch_sequence_lengths[output_name][utt_ind], :]

			# handle the output
			handled_output = self.handle_output(utt_output, utt_name)

			# store the handled output
			filename = os.path.join(self.store_dir, 'data', utt_name + '.npy')
			np.save(filename, handled_output)

			# if requested store the raw output
			if self.store_output:
				for output_name in self.requested_output_names:
					savename = output_name+'_'+ utt_name
					np.save(os.path.join(self.raw_output_dir, savename), utt_output[output_name])

			self.pos += 1


	@abstractmethod
	def handle_output(self, output, utt_name):
		"""handle the output

		Args:
			output: the output of a single utterance of the neural network
			utt_name: the name of the utterance

		Returns:
			the handled output"""

	def open_scp_files(self, from_start=True):
		pass
