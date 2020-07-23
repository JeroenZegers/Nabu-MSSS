"""@file scorer.py
contains the Scorer class"""

from abc import ABCMeta, abstractmethod
import os
import numpy as np
from nabu.postprocessing import data_reader


class Scorer(object):
	""""""
	def __init__(self, conf, evalconf, dataconf, store_dir, numbatches, task, scorer_name, checkpoint_file):
		"""Scorer constructor

		Args:
			conf: the scorer configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			store_dir: the directory where the reconstructions are
			numbatches: the number of batches to process
		"""

		if evalconf.has_option(task, 'batch_size'):
			batch_size = int(evalconf.get(task, 'batch_size'))
		else:
			batch_size = int(evalconf.get('evaluator', 'batch_size'))
		self.tot_utt = batch_size * numbatches
		self.store_dir = store_dir
		self.segment_lengths = evalconf.get('evaluator', 'segment_length').split(' ')

		# get the feature input reader, only to get the name of the utterance actually.
		input_features_names = conf['input_features'].split(' ')
		input_features_dataconfs = []
		for input_features_name in input_features_names:
			input_features_dataconfs.append(dict(dataconf.items(input_features_name)))
		self.input_features_reader = data_reader.DataReader(input_features_dataconfs, self.segment_lengths)

		if 'nrs' in conf:
			self.nrS = int(conf['nrs'])

		# create the dictionary where all results will be stored
		self.results = dict()

		self.pos = 0
		self.scorer_name = scorer_name

	def __call__(self):
		""" score the utterances in the reconstruction dir with the original source signals

		"""

		for utt_ind in range(self.tot_utt):
			if np.mod(utt_ind, 100) == 0:
				print 'Getting results for utterance %d' % utt_ind

			utt_name = self.input_features_reader.get_name_for_pos(self.pos)

			handled_output_filename = os.path.join(self.store_dir, 'data', utt_name + '.npy')
			handled_output = np.load(handled_output_filename)

			[enr_vecs, target_labels] = self.get_enr_vecs(utt_name)

			utt_score_dict = self._get_score(handled_output, enr_vecs, target_labels)

			# assemble results
			self.results[utt_name] = dict()
			self.results[utt_name]['score'] = utt_score_dict
			# self.results[utt_name]['spk_info'] = spk_info

			self.pos += 1

	@abstractmethod
	def summarize(self):
		"""summarize the results of all utterances

		"""

	@abstractmethod
	def _get_score(self, handled_output, enr_vecs, target_labels):
		"""score the handled output.

		Args:
			handled_output
			enr_vecs
			target_labels

		Returns:
			the score"""

	@abstractmethod
	def get_enr_vecs(self, utt_ind):
		"""Get the enrollment vector.

		Args:
			utt_ind

		Returns:
			enr_vecs: the enrollment i-vectors to compare with
			target_labels: whether the enrolment vector is from the same speaker as the test vector"""
