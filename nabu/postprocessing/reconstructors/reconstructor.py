"""@file reconstructor.py
contains the Reconstructor class"""

from abc import ABCMeta, abstractmethod
import os
import scipy.io.wavfile as wav
import numpy as np


class Reconstructor(object):
	"""the general reconstructor class

	a reconstructor is used to reconstruct the signals from the models output"""

	__metaclass__ = ABCMeta

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""Reconstructor constructor

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
		self.optimal_frame_permutation = optimal_frame_permutation

		self.nrS = int(conf['nrs'])

		if 'transpose_order' in conf:
			self.transpose_order = map(int, conf['transpose_order'].split(' '))
		else:
			self.transpose_order = False

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

		self.scp_file = os.path.join(self.rec_dir, 'pointers.scp')

		# whether to save output as numpy instead of wav file
		if 'save_as_numpy' in conf:
			self.save_as_numpy = conf['save_as_numpy'] in ['True', 'true']
		else:
			self.save_as_numpy = False

		# Whether the raw output should also be stored (besides the reconstructed audiosignal)
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
		if self.transpose_order:
			for output_name in self.requested_output_names:
				batch_outputs[output_name] = np.transpose(batch_outputs[output_name], self.transpose_order)

		for utt_ind in range(self.batch_size):

			utt_output = dict()
			for output_name in self.requested_output_names:
				# anchor output for anchor_deepattractornet_softmax_reconstructor is special case
				if output_name is 'anchors' and self.__class__.__name__ in ['AnchorDeepattractorSoftmaxReconstructor', 'WeightedAnchorDeepattractorSoftmaxReconstructor']:
					utt_output[output_name] = batch_outputs[output_name]
				elif output_name is 'anchors_scale' and self.__class__.__name__ in ['TimeAnchorScalarDeepattractorSoftmaxReconstructor']:
					utt_output[output_name] = batch_outputs[output_name]
				else:
					utt_output[output_name] = \
						batch_outputs[output_name][utt_ind][:batch_sequence_lengths[output_name][utt_ind], :]

			# reconstruct the signals
			reconstructed_signals, utt_info = self.reconstruct_signals(utt_output)

			# make the audio files for the reconstructed signals
			if self.save_as_numpy:
				filename = os.path.join(self.rec_dir, utt_info['utt_name'] + '.npy')
				np.save(filename, reconstructed_signals)
			else:
				self.write_audiofile(reconstructed_signals, utt_info)

			# if requested store the raw output
			if self.store_output:
				for output_name in self.requested_output_names:
					savename = output_name+'_'+utt_info['utt_name']
					np.save(os.path.join(self.output_dir, savename), utt_output[output_name])

			self.pos += 1

	def opt_frame_perm(self, batch_outputs, batch_targets, batch_sequence_lengths):
		""" reconstruct the signals, using the optimal speaker permutations on frame level using the targets, and write
		the audio files

		Args:
		- batch_outputs: A dictionary containing the batch outputs of the network
		- batch_outputs: A dictionary containing the batch targets for the outputs
		- batch_sequence_lengths: A dictionary containing the sequence length for each utterance
		"""

		for utt_ind in range(self.batch_size):

			utt_output = dict()
			for output_name in self.requested_output_names:
				utt_output[output_name] = \
					batch_outputs[output_name][utt_ind][:batch_sequence_lengths[output_name][utt_ind], :]
			# assuming only one requested target
			target_keys = [key for key in batch_targets.keys() if 'target' in key]
			utt_target = {
				key: batch_targets[key][utt_ind][:batch_sequence_lengths[output_name][utt_ind], :]
				for key in target_keys}

			# reconstruct the signals
			reconstructed_signals, utt_info = self.reconstruct_signals_opt_frame_perm(utt_output, utt_target)

			# make the audio files for the reconstructed signals
			self.write_audiofile(reconstructed_signals, utt_info)

			# if requested store the raw output
			if self.store_output:
				for output_name in self.requested_output_names:
					savename = output_name+'_'+utt_info['utt_name']
					np.save(os.path.join(self.output_dir, savename), utt_output[output_name])

			self.pos += 1

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

		write_str = utt_info['utt_name']
		for spk in range(self.nrS):
			rec_dir = os.path.join(self.rec_dir, 's' + str(spk+1))
			filename = os.path.join(rec_dir, utt_info['utt_name']+'.wav')
			signal = reconstructed_signals[spk]
			if signal.dtype == np.float64:
				signal = np.float32(signal)
			wav.write(filename, utt_info['rate'], signal)
			write_str += ' ' + filename

		write_str += ' \n'
		self.scp_fid.write(write_str)

	def open_scp_files(self, from_start=True):
		if from_start:
			file_mode = 'w'
		else:
			file_mode = 'a+'
		self.scp_fid = open(self.scp_file, file_mode)

