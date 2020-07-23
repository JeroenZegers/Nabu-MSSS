"""@file scorer.py
contains the Scorer class"""

from abc import ABCMeta, abstractmethod
import os
import scipy.io.wavfile as wav
from nabu.postprocessing import data_reader
import numpy as np
import json


class Scorer(object):
	"""the general scorer class

	a scorer is used to score the reconstructed signals"""

	__metaclass__ = ABCMeta
	target_name = 'org_src'
	statistics_to_summarize = ['mean', 'std', 'median']

	def __init__(self, conf, evalconf, dataconf, rec_dir, numbatches, task, scorer_name, checkpoint_file):
		"""Scorer constructor

		Args:
			conf: the scorer configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions are
			numbatches: the number of batches to process
		"""

		if evalconf.has_option(task, 'batch_size'):
			batch_size = int(evalconf.get(task, 'batch_size'))
		else:
			batch_size = int(evalconf.get('evaluator', 'batch_size'))
		self.tot_utt = batch_size * numbatches
		self.rec_dir = rec_dir
		self.segment_lengths = evalconf.get('evaluator', 'segment_length').split(' ')

		# get the target reader (typically the original source signals)
		target_names = conf[self.target_name].split(' ')
		target_dataconfs = []
		for target_name in target_names:
			target_dataconfs.append(dict(dataconf.items(target_name)))
		self.target_reader = data_reader.DataReader(target_dataconfs, self.segment_lengths)

		# get the base signal (original mixture) reader
		base_names = conf['base'].split(' ')
		base_dataconfs = []
		for base_name in base_names:
			base_dataconfs.append(dict(dataconf.items(base_name)))
		self.base_reader = data_reader.DataReader(base_dataconfs, self.segment_lengths)

		if 'select_indices' in conf:
			select_indices_names = conf['select_indices'].split(' ')
			select_indices_dataconfs = []
			for select_indices_name in select_indices_names:
				select_indices_dataconfs.append(dict(dataconf.items(select_indices_name)))
			self.select_indices_reader = data_reader.DataReader(select_indices_dataconfs, self.segment_lengths)
		else:
			self.select_indices_reader = False

		if 'nrs' in conf:
			self.nrS = int(conf['nrs'])

		# get the speaker info
		self.utt_spkinfo = dict()
		spkinfo_names = conf['spkinfo'].split(' ')
		for spkinfo_name in spkinfo_names:
			spkinfo_dataconf = dict(dataconf.items(spkinfo_name))
			spkinfo_file = spkinfo_dataconf['datafiles']

			for line in open(spkinfo_file):
				splitline = line.strip().split(' ')
				utt_name = splitline[0]
				dataline = ' '.join(splitline[2:])
				self.utt_spkinfo[utt_name] = dataline
		# predefined mixture types
		self.mix_types = ['all_m', 'all_f', 'same_gen', 'diff_gen']
		# metrics to be used in sumarize function, if not yet stated
		if not self.score_metrics_to_summarize:
			self.score_metrics_to_summarize = self.score_metrics

		if 'score_from' in conf and conf['score_from'] == 'True':
			if self.score_expects != 'data':
				raise Exception('Can only score from a specific timestamp on if scorer expects data')
			else:
				self.score_from = True
		else:
			self.score_from = False

		if 'score_center_samples_num' in conf:
			if self.score_expects != 'data':
				raise BaseException('')
			else:
				self.score_center_samples_num = int(conf['score_center_samples_num'])
				self.score_center_samples_num = float(self.score_center_samples_num)
		else:
			self.score_center_samples_num = False

		if 'load_rec_as_numpy' in conf:
			self.load_rec_as_numpy = conf['load_rec_as_numpy'] in ['True', 'true']
		else:
			self.load_rec_as_numpy = False

		# create the dictionary where all results will be stored
		self.results = dict()

		self.checkpoint_file = checkpoint_file
		if self.checkpoint_file:
			if os.path.isfile(self.checkpoint_file):
				with open(self.checkpoint_file, 'r') as fid:
					self.results = json.load(fid)
					self.start_ind = len(self.results.keys())
			else:
				self.start_ind = 0
		else:
			self.start_ind = 0

	def __call__(self):
		""" score the utterances in the reconstruction dir with the original source signals

		"""

		for utt_ind in range(self.start_ind, self.tot_utt):
			if np.mod(utt_ind, 10) == 0:
				print 'Getting results for utterance %d' % utt_ind

			if self.score_expects == 'data':
				# Gather the data for scoring

				# get the source signals
				target_signals, utt_info = self.target_reader(utt_ind)
				if hasattr(self, 'nrS'):
					nrS = self.nrS
				else:
					nrS = utt_info['nrSig']
				utt_name = utt_info['utt_name']

				# determine from which sample on to score. This should be mentioned at the
				# end of the utt_name
				if self.score_from:
					score_from_sample = int(utt_name.split('_')[-1])
					target_signals = target_signals[:, score_from_sample+1:]

				# get the base signal (original mixture) and duplicate it
				base_signal, _ = self.base_reader(utt_ind)
				if self.score_from:
					base_signal = base_signal[score_from_sample+1:]
				base_signals = list()
				for spk in range(nrS):
					base_signals.append(base_signal)

				if self.select_indices_reader:
					active_spk, _ = self.select_indices_reader(utt_ind)
				else:
					active_spk = range(nrS)

				# get the reconstructed signals
				if self.load_rec_as_numpy:
					rec_src_filenames = os.path.join(self.rec_dir, utt_name+'.npy')
					rec_src_signals = np.load(rec_src_filenames)
				else:
					rec_src_signals = list()
					rec_src_filenames = list()
					for spk in active_spk:
						filename = os.path.join(self.rec_dir, 's'+str(spk+1), utt_name+'.wav')
						_, utterance = wav.read(filename)
						if self.score_from:
							utterance = utterance[score_from_sample+1:]
						rec_src_signals.append(utterance)
						rec_src_filenames.append(filename)

				if self.score_center_samples_num:
					num_samples = float(len(target_signals[0]))
					st_sample_to_score = int(round(num_samples/2-self.score_center_samples_num/2))
					st_sample_to_score = np.max([st_sample_to_score, 0])
					end_sample_to_score = int(round(num_samples/2+self.score_center_samples_num/2))
					end_sample_to_score = np.min([end_sample_to_score, int(num_samples)])
					samples_to_score = range(st_sample_to_score, end_sample_to_score)
					target_signals = target_signals[:, samples_to_score, :]
					base_signals = [s[samples_to_score, :] for s in base_signals]
					rec_src_signals = [s[samples_to_score] for s in rec_src_signals]

				if not hasattr(self, 'noise_reader'):
					# get the scores for the utterance (in dictionary format)
					utt_score_dict = self._get_score(target_signals, base_signals, rec_src_signals, utt_rate=utt_info['rate'])
				else:
					noise_signal, _ = self.noise_reader(utt_ind)
					if self.score_from:
						noise_signal = noise_signal[score_from_sample+1:]
					utt_score_dict = self._get_score(target_signals, base_signals, rec_src_signals, noise_signal)

			elif self.score_expects == 'files':
				# Gather the filnames for scoring

				splitline = self.target_reader.datafile_lines[utt_ind].strip().split(' ')
				utt_name = splitline[0]
				target_filenames = splitline[1:]
				if hasattr(self, 'nrS'):
					nrS = self.nrS
				else:
					nrS = len(target_filenames)

				splitline = self.base_reader.datafile_lines[utt_ind].strip().split(' ')
				base_filename = splitline[1]
				base_filenames = list()
				for spk in range(nrS):
					base_filenames.append(base_filename)

				rec_src_filenames = list()
				for spk in range(nrS):
					filename = os.path.join(self.rec_dir, 's'+str(spk+1), utt_name+'.wav')
					rec_src_filenames.append(filename)

				if not hasattr(self, 'noise_reader'):
					# get the scores for the utterance (in dictionary format)
					utt_score_dict = self._get_score(target_filenames, base_filenames, rec_src_filenames)
				else:
					splitline = self.noise_reader.datafile_lines[utt_ind].strip().split(' ')
					noise_filename = splitline[1]
					utt_score_dict = self._get_score(target_filenames, base_filenames, rec_src_filenames, noise_filename)

			else:
				raise Exception('unexpected input for scrorer_expects: %s' % self.score_expects)

			# get the speaker info
			spk_info = dict()
			spk_info['ids'] = []
			spk_info['genders'] = []
			dataline = self.utt_spkinfo[utt_name]
			splitline = dataline.strip().split(' ')
			for spk in range(nrS):
				spk_info['ids'].append(splitline[spk*2])
				spk_info['genders'].append(splitline[spk*2+1])

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

			# if len(utt_score_dict[self.score_metrics_to_summarize[0]][self.score_scenarios[0]]) != nrS:
			# 	raise BaseException('Got unexpected number of speakers')

			if self.checkpoint_file and np.mod(utt_ind, 1000) == 0:
				print 'Checkpointing results at utt %d' % utt_ind
				with open(self.checkpoint_file, 'w') as fid:
					json.dump(self.results, fid)

	def summarize(self):
		"""summarize the results of all utterances

		"""
		#
		utts = self.results.keys()

		mix_type_indeces = dict()
		for mix_type in self.mix_types:
			mix_type_indeces[mix_type] = []

			for i, utt in enumerate(utts):
				if self.results[utt]['spk_info']['mix_type'][mix_type]:
					mix_type_indeces[mix_type].append(i)

		result_summary = dict()
		for metric in self.score_metrics_to_summarize:
			result_summary[metric] = dict()

			for scen in self.score_scenarios:
				result_summary[metric][scen] = dict()

				tmp = []
				for i, utt in enumerate(utts):

					utt_score = np.mean(self.results[utt]['score'][metric][scen])
					tmp.append(utt_score)

				result_summary[metric][scen]['all'] = get_statistics(tmp, self.statistics_to_summarize)
				for mix_type in self.mix_types:
					inds = mix_type_indeces[mix_type]
					result_summary[metric][scen][mix_type] = get_statistics(
						[tmp[i] for i in inds], self.statistics_to_summarize)
		#
		for metric in self.score_metrics_to_summarize:
			print ''
			print 'Result for %s (using %s): ' % (metric, self.__class__.__name__)

			for mix_type in ['all']+self.mix_types:
				print 'for %s: ' % mix_type
				
				for stat in self.statistics_to_summarize:
					print '\t %s: ' % stat,
					for scen in self.score_scenarios:
						print '%f (%s), ' % (result_summary[metric][scen][mix_type][stat], scen),
					# if only 2 scenarios, print the difference
					if len(self.score_scenarios) == 2:
						scen1 = self.score_scenarios[0]
						scen2 = self.score_scenarios[1]
						diff = result_summary[metric][scen1][mix_type][stat] - result_summary[metric][scen2][mix_type][stat]
						print '%f (absolute difference)' % diff
					else:
						print ''

		return result_summary

	# @abstractmethod
	# def _get_score(self, target_signals, base_signals, rec_src_signals, *args):
	# 	"""score the reconstructed utterances with respect to the original source signals.
	# 	This score should be independent to permutations.
	#
	# 	Args:
	# 		target_signals: the original source signals, as a list of numpy arrays. May also be a list of audio
	# 			filenames
	# 		base_signals: the duplicated base signal (original mixture), as a list of numpy arrays. May also be a list
	# 			of audio filenames
	# 		rec_src_signals: the reconstructed source signals, as a list of numpy arrays. May also be a list of audio
	# 			filenames
	# 		args: option aditonal signals, as a numpy arrays. May also be an audio filenames
	#
	# 	Returns:
	# 		the score"""

	# @abstractproperty
	# def score_metrics():

	# @abstractproperty
	# def score_scenarios():

	# @abstractproperty
	# def score_expects():

	def storable_result(self):
		return self.results

# TODO: make the ScroreTogether class, which allows to score reconstructions together.


def get_statistics(data, statistics):
	statistics_dict = dict()
	for stat in statistics:
		if stat in ['mean', 'average']:
			statistics_dict[stat] = np.mean(data)
		elif stat in ['std', 'deviation', 'standard_deviation']:
			statistics_dict[stat] = np.std(data)
		elif stat in ['var', 'variance']:
			statistics_dict[stat] = np.var(data)
		elif stat == 'median':
			statistics_dict[stat] = np.median(data)
		else:
			raise BaseException('Unknown statistic: %s', stat)
	
	return statistics_dict
