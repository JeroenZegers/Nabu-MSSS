"""@file sdr_snr_noise_scorer.py
contains the scorer using SdrSnrNoiseScorer"""
# Edited by Pieter Appeltans (added snr score)
import scorer
import numpy as np
import os
import scipy.io.wavfile as wav
from nabu.postprocessing import data_reader
import bss_eval


class SdrSnrNoiseScorer(scorer.Scorer):
	"""the SDR scorer class. Uses the script from
	C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis,
	'mir_eval: A Transparent Implementation of Common MIR Metrics', Proceedings of the 15th
	International Conference on Music Information Retrieval, 2014

	a scorer using SDR

	"""

	score_metrics = ('SDR', 'SIR', 'SNR', 'SAR', 'perm')
	score_metrics_to_summarize = ('SDR', 'SIR', 'SNR', 'SAR')
	score_scenarios = ('SS', 'base')
	score_expects = 'data'

	def __init__(self, conf, evalconf, dataconf, rec_dir, numbatches, task, scorer_name, checkpoint_file):
		"""Reconstructor constructor
		Args:
			conf: the scorer configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions are
			numbatches: the number of batches to process
		"""

		super(SdrSnrNoiseScorer, self).__init__(conf, evalconf, dataconf, rec_dir, numbatches, task, scorer_name, checkpoint_file)

		# get the original noise signal reader
		noise_names = conf['noise'].split(' ')
		noise_dataconfs = []
		for noise_name in noise_names:
			noise_dataconfs.append(dict(dataconf.items(noise_name)))
		self.noise_reader = data_reader.DataReader(noise_dataconfs, self.segment_lengths)

	def __call__(self):
		""" score the utterances in the reconstruction dir with the original source signals

		"""
		for utt_ind in range(self.tot_utt):
			if np.mod(utt_ind, 10) == 0:
				print 'Getting results for utterance %d' % utt_ind

			if self.score_expects == 'data':
				# Gather the data for scoring

				# get the source signals
				org_src_signals, utt_info = self.org_src_reader(utt_ind)
				nrS = utt_info['nrSig']
				nrC = nrS+1
				utt_name = utt_info['utt_name']

				# determine from which sample on to score. This should be mentioned at the
				# end of the utt_name
				if self.score_from:
					score_from_sample = int(utt_name.split('_')[-1])
					org_src_signals = org_src_signals[:, score_from_sample + 1:]

				# get the base signal (original mixture) and duplicate it
				base_signal, _ = self.base_reader(utt_ind)
				if self.score_from:
					base_signal = base_signal[score_from_sample + 1:]
				base_signals = list()
				for spk in range(nrS):
					base_signals.append(base_signal)

				# get the reconstructed signals
				rec_src_signals = list()
				rec_src_filenames = list()
				for spk in range(nrC):
					filename = os.path.join(self.rec_dir, 's' + str(spk + 1), utt_name + '.wav')
					_, utterance = wav.read(filename)
					if self.score_from:
						utterance = utterance[score_from_sample + 1:]
					rec_src_signals.append(utterance)
					rec_src_filenames.append(filename)

				noise_signal, _ = self.noise_reader(utt_ind)
				if self.score_from:
					noise_signal = noise_signal[score_from_sample + 1:]

				utt_score_dict = self._get_score(org_src_signals, base_signals, rec_src_signals, noise_signal)

			elif self.score_expects == 'files':
				# Gather the filnames for scoring

				splitline = self.org_src_reader.datafile_lines[utt_ind].strip().split(' ')
				utt_name = splitline[0]
				org_src_filenames = splitline[1:]
				nrS = len(org_src_filenames)
				nrC = nrS+1

				splitline = self.base_reader.datafile_lines[utt_ind].strip().split(' ')
				base_filename = splitline[1]
				base_filenames = list()
				for spk in range(nrS):
					base_filenames.append(base_filename)

				rec_src_filenames = list()
				for spk in range(nrC):
					filename = os.path.join(self.rec_dir, 's' + str(spk + 1), utt_name + '.wav')
					rec_src_filenames.append(filename)

				splitline = self.noise_reader.datafile_lines[utt_ind].strip().split(' ')
				noise_filename = splitline[1]
				utt_score_dict = self._get_score(org_src_filenames, base_filenames, rec_src_filenames, noise_filename)

			else:
				raise Exception('unexpected input for scrorer_expects: %s' % self.score_expects)

			# get the speaker info
			spk_info = dict()
			spk_info['ids'] = []
			spk_info['genders'] = []
			dataline = self.utt_spkinfo[utt_name]
			splitline = dataline.strip().split(' ')
			for spk in range(nrS):
				spk_info['ids'].append(splitline[spk * 2])
				spk_info['genders'].append(splitline[spk * 2 + 1])

			spk_info['mix_type'] = dict()
			for mix_type in self.mix_types:
				spk_info['mix_type'][mix_type] = False
			if all(gender == 'M' for gender in spk_info['genders']):
				spk_info['mix_type']['all_m'] = True
				spk_info['mix_type']['same_gen'] = True
			elif all(gender == 'F' for gender in spk_info['genders']):
				spk_info['mix_type']['all_f'] = True
				spk_info['mix_type']['same_gen'] = True
			else:
				spk_info['mix_type']['diff_gen'] = True

			# assemble results
			self.results[utt_name] = dict()
			self.results[utt_name]['score'] = utt_score_dict
			self.results[utt_name]['spk_info'] = spk_info

	def _get_score(self, org_src_signals, base_signals, rec_src_signals, noise_signal):
		"""score the reconstructed utterances with respect to the original source signals

		Args:
			org_src_signals: the original source signals, as a list of numpy arrarys
			base_signals: the duplicated base signal (original mixture), as a list of numpy arrarys
			rec_src_signals: the reconstructed source signals, as a list of numpy arrarys

		Returns:
			the score"""

		# convert to numpy arrays
		org_src_signals = np.array(org_src_signals)[:, :, 0]
		base_signals = np.array(base_signals)[:, :, 0]
		rec_src_signals = np.array(rec_src_signals)
		noise_signal = np.squeeze(noise_signal)
		#
		collect_outputs = dict()
		collect_outputs[self.score_scenarios[1]] = bss_eval.bss_eval_sources_extended(org_src_signals, base_signals, noise_signal)
		collect_outputs[self.score_scenarios[0]] = bss_eval.bss_eval_sources_extended_noise(org_src_signals, rec_src_signals, noise_signal)

		nr_spk = len(org_src_signals)

		# convert the outputs to a single dictionary
		score_dict = dict()
		for i, metric in enumerate(self.score_metrics):
			score_dict[metric] = dict()

			for j, scen in enumerate(self.score_scenarios):
				score_dict[metric][scen] = []

				for spk in range(nr_spk):
					score_dict[metric][scen].append(collect_outputs[scen][i][spk])

		return score_dict
