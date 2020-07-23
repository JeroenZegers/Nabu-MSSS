"""@file equal_error_rate_from_vecs.py
contains the EERFromIvecs class"""

import os
import numpy as np
import scorer
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import json


class EERFromIvecs(scorer.Scorer):
	""""""
	def __init__(self, conf, evalconf, dataconf, store_dir, numbatches, task, scorer_name, checkpoint_file):
		"""EER constructor

		Args:
			conf: the scorer configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			store_dir: the directory where the reconstructions are
			numbatches: the number of batches to process
		"""
		super(EERFromIvecs, self).__init__(conf, evalconf, dataconf, store_dir, numbatches, task, scorer_name, checkpoint_file)

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

		self.num_trials_per_mix_per_spk = int(conf['num_trials_per_mix_per_spk'])

		# Normalizing a speakers score for all its enrollments using mvn. It is
		# acceptable if only using non-target enrollments for the normalization
		self.mvn_score_per_mix_spk = False
		if 'mvn_score_per_mix_spk' in conf and conf['mvn_score_per_mix_spk'] != 'False':
			if conf['mvn_score_per_mix_spk'] in ['True', 'all']:
				self.mvn_score_per_mix_spk = 'all'
			elif conf['mvn_score_per_mix_spk'] == 'non-target only':
				self.mvn_score_per_mix_spk = 'non-target only'
			else:
				raise BaseException('Unknown option %s for mvn_score_per_mix_spk' % conf['mvn_score_per_mix_spk'])

		thr_range_parse = map(float, conf['thr_range'].split(' '))
		self.thr_range = np.arange(thr_range_parse[0], thr_range_parse[1], thr_range_parse[2])

		if 'make_figure' in conf and conf['make_figure'] == 'True':
			self.make_figure = True
		else:
			self.make_figure = False

		if 'load_trial_list' in conf and not conf['load_trial_list'] == 'None':
			load_trial_list = conf['load_trial_list']
		else:
			load_trial_list = None

		if 'gender_specific_trials' in conf:
			gender_specific_trials = conf['gender_specific_trials']
		else:
			gender_specific_trials = 'all'

		only_competing_spk = 'only_competing_spk' in conf and conf['only_competing_spk'] == 'True'

		store_trial_list = not load_trial_list and ('store_trial_list' not in conf or conf['store_trial_list'] == 'True')
		# store_trial_list = not load_trial_list and ('store_trial_list' in conf and conf['store_trial_list'] == 'True')

		if 'load_enr_vecs' in conf and not conf['load_enr_vecs'] == 'None':
			load_enr_vecs = conf['load_enr_vecs']
		else:
			load_enr_vecs = None

		self.all_enr_vecs, self.all_enr_vec_pointers, self.all_target_labels = self.create_enr_vecs(
			store_dir, self.utt_spkinfo, gender_specific_trials, only_competing_spk, load_trial_list, load_enr_vecs)

		if store_trial_list:
			trial_list_file_to_store = os.path.join(store_dir, 'trials_%s.json' % scorer_name)
			with open(trial_list_file_to_store, 'w') as fid:
				all_target_labels_json = {key: [np.array(spk_val, dtype=int) for spk_val in val] for key, val in self.all_target_labels.iteritems()}
				all_target_labels_json = {key: [[tmp for tmp in spk_val] for spk_val in val] for key, val in all_target_labels_json.iteritems()}
				json.dump({'enr_vec_pointers': self.all_enr_vec_pointers, 'target_labels': all_target_labels_json}, fid)

	def create_enr_vecs(
			self, store_dir, utt_spkinfo, gender_specific_trials='all', only_competing_spk=False, load_trial_list=None,
			load_enr_vecs=None):
		all_utt_names = utt_spkinfo.keys()

		# First gather all handled outputs
		if not load_enr_vecs:
			load_enr_vecs = store_dir

		all_outputs = dict()
		for utt_name in all_utt_names:
			handled_output_filename = os.path.join(load_enr_vecs, 'data', utt_name + '.npy')
			handled_output = np.load(handled_output_filename)
			all_outputs[utt_name] = handled_output

		#
		if load_trial_list:
			with open(load_trial_list, 'r') as fid:
				json_dump = json.load(fid)
				all_enr_vec_pointers = json_dump['enr_vec_pointers']
				all_target_labels = json_dump['target_labels']
				all_target_labels = {key: np.array(val, dtype=bool) for key, val in all_target_labels.iteritems()}
		else:
			mix2spk = dict()
			spk2mix = dict()
			spk2gender = dict()
			for utt_name in all_utt_names:
				dataline_split = utt_spkinfo[utt_name].strip('\n').split(' ')
				mix_spk_ids = dataline_split[::2]
				mix_spk_genders = dataline_split[1::2]
				mix2spk[utt_name] = mix_spk_ids
				for spk_ind, spk_id in enumerate(mix_spk_ids):
					if spk_id in spk2mix:
						spk2mix[spk_id].append([utt_name, spk_ind])
					else:
						spk2mix[spk_id] = [[utt_name, spk_ind]]
					if spk_id not in spk2gender.keys():
						spk2gender[spk_id] = mix_spk_genders[spk_ind]
			all_spks = spk2mix.keys()

			# for each speaker in each mixture, make target and non-target trials
			all_enr_vec_pointers = dict()
			all_target_labels = dict()
			for utt_name in all_utt_names:
				mix_spk_ids = mix2spk[utt_name]
				other_spk = list(set(all_spks) - set(mix_spk_ids))

				all_enr_vec_pointers_mix = []
				all_target_labels_mix = []

				for spk_id in mix_spk_ids:
					spk_gender = spk2gender[spk_id]
					if (spk_gender == 'F' and gender_specific_trials == 'only_m') or (spk_gender == 'M' and gender_specific_trials == 'only_f'):
						all_enr_vec_pointers_mix.append([])
						all_target_labels_mix.append([])
					else:
						# positive enrollments
						utt_names_and_spk_inds_spk = spk2mix[spk_id]
						utt_names_and_spk_inds_spk = [tmp for tmp in utt_names_and_spk_inds_spk if tmp[0] != utt_name]
						num_options = len(utt_names_and_spk_inds_spk)
						if num_options < self.num_trials_per_mix_per_spk:
							pos_enr_pointers = utt_names_and_spk_inds_spk
							pos_target_labels = np.ones(num_options, dtype=bool)
						else:
							pos_enr_pointers = random.sample(utt_names_and_spk_inds_spk, self.num_trials_per_mix_per_spk)
							pos_target_labels = np.ones(self.num_trials_per_mix_per_spk, dtype=bool)

						# negative enrollments
						neg_enr_pointers = []
						if not only_competing_spk:
							while len(neg_enr_pointers) < self.num_trials_per_mix_per_spk:
								# First, find a random enrollment, then check if it is a valid one.
								random_utt_name = random.choice(all_utt_names)
								random_spk_ind = random.randint(0, self.nrS-1)
								spk_ids_of_random_utt = mix2spk[random_utt_name]
								spk_id_of_random_utt = spk_ids_of_random_utt[random_spk_ind]

								valid_random = True
								valid_random = valid_random and all([random_spk_id in other_spk for random_spk_id in spk_ids_of_random_utt])
								random_spk_gender = spk2gender[spk_id_of_random_utt]
								valid_random = valid_random and not \
									(random_spk_gender == 'F' and gender_specific_trials == 'only_m') and not \
									(random_spk_gender == 'M' and gender_specific_trials == 'only_f')
								valid_random = valid_random and not \
									(gender_specific_trials in ['same_gen', 'same_gender'] and spk_gender != random_spk_gender)
								valid_random = valid_random and not \
									(gender_specific_trials in ['diff_gen', 'diff_gender'] and spk_gender == random_spk_gender)
								if valid_random:
									neg_enr_pointers.append([random_utt_name, random_spk_ind])
						else:
							competing_spk_ids = list(set(mix_spk_ids) - set([spk_id]))
							available_mix_pointers = [mix_pointer for competing_id in competing_spk_ids for mix_pointer in spk2mix[competing_id]]
							random.shuffle(available_mix_pointers)
							for mix_pointer in available_mix_pointers:
								valid_random = True
								spk_id_of_random_utt = mix2spk[mix_pointer[0]][mix_pointer[1]]
								random_spk_gender = spk2gender[spk_id_of_random_utt]
								valid_random = valid_random and not \
									(random_spk_gender == 'F' and gender_specific_trials == 'only_m') and not \
									(random_spk_gender == 'M' and gender_specific_trials == 'only_f')
								valid_random = valid_random and not \
									(gender_specific_trials in ['same_gen', 'same_gender'] and spk_gender != random_spk_gender)
								valid_random = valid_random and not \
									(gender_specific_trials in ['diff_gen', 'diff_gender'] and spk_gender == random_spk_gender)
								if valid_random:
									neg_enr_pointers.append(mix_pointer)
									if len(neg_enr_pointers) >= self.num_trials_per_mix_per_spk:
										break

						neg_target_labels = np.zeros(len(neg_enr_pointers), dtype=bool)

						all_enr_vec_pointers_mix_spk = pos_enr_pointers + neg_enr_pointers
						all_target_labels_mix_spk = np.concatenate([pos_target_labels, neg_target_labels])

						all_enr_vec_pointers_mix.append(all_enr_vec_pointers_mix_spk)
						all_target_labels_mix.append(all_target_labels_mix_spk)

				all_enr_vec_pointers[utt_name] = all_enr_vec_pointers_mix
				all_target_labels[utt_name] = all_target_labels_mix

		return all_outputs, all_enr_vec_pointers, all_target_labels

	def _get_score(self, handled_output, enr_vecs, labels):
		"""score the handled output.

		Args:
			handled_output

		Returns:
			the score"""

		score_dict = list()
		for spk_ind in range(self.nrS):
			handled_output_spk = handled_output[spk_ind]
			enr_vecs_spk = enr_vecs[spk_ind]
			labels_spk = labels[spk_ind]
			score_dict.append(dict())
			if len(enr_vecs_spk) == 0:
				score_dict[spk_ind]['binary'] = []
				score_dict[spk_ind]['target'] = []
				score_dict[spk_ind]['false'] = []

			else:
				cos_sims = np.dot(enr_vecs_spk, handled_output_spk)
				if self.mvn_score_per_mix_spk:
					if self.mvn_score_per_mix_spk == 'all':
						mean = np.mean(cos_sims)
						std = np.std(cos_sims)
					elif self.mvn_score_per_mix_spk == 'non-target only':
						mean = np.mean(cos_sims[labels_spk != 1])
						std = np.std(cos_sims[labels_spk != 1])
					else:
						raise BaseException('Unknown score mvn type %s' % self.mvn_score_per_mix_spk)
					cos_sims = cos_sims - mean
					cos_sims = cos_sims / std

				score_to_binary_spk = [[cos_sim > thr for thr in self.thr_range] for cos_sim in cos_sims]

				score_dict[spk_ind]['binary'] = score_to_binary_spk
				score_dict[spk_ind]['target'] = labels_spk

				correct = np.equal(score_to_binary_spk, np.expand_dims(labels_spk, -1))
				score_dict[spk_ind]['false'] = np.logical_not(correct)

		return score_dict

	def get_enr_vecs(self, utt_name):
		mix_enr_vec_pointers = self.all_enr_vec_pointers[utt_name]
		mix_enr_vecs = []
		for spk_ind in range(len(mix_enr_vec_pointers)):
			mix_spk_enr_vecs = []
			for poin in mix_enr_vec_pointers[spk_ind]:
				mix_spk_enr_vecs.append(self.all_enr_vecs[poin[0]][poin[1]])
			mix_enr_vecs.append(mix_spk_enr_vecs)

		return mix_enr_vecs,  self.all_target_labels[utt_name]

	def storable_result(self):
		storable_result = copy.deepcopy(self.results)
		for utt in storable_result:
			for spk_ind in range(len(storable_result[utt]['score'])):
				storable_result[utt]['score'][spk_ind]['binary'] = [map(int, tmp) for tmp in storable_result[utt]['score'][spk_ind]['binary']]
				storable_result[utt]['score'][spk_ind]['false'] = [map(int, tmp) for tmp in storable_result[utt]['score'][spk_ind]['false']]
				storable_result[utt]['score'][spk_ind]['target'] = map(int, storable_result[utt]['score'][spk_ind]['target'])

	def summarize(self):
		"""summarize the results of all utterances

		"""
		#
		utts = self.results.keys()

		num_ground_trues = 0
		num_ground_falses = 0
		num_false_negative = np.zeros(np.shape(self.thr_range))
		num_false_positive = np.zeros(np.shape(self.thr_range))
		for utt in utts:
			utt_result = self.results[utt]

			for spk_ind in range(self.nrS):
				if np.size(utt_result['score'][spk_ind]['target']) == 1:
					utt_result['score'][spk_ind]['target'] = [utt_result['score'][spk_ind]['target']]
				for tar_ind, tar in enumerate(utt_result['score'][spk_ind]['target']):
					if tar:
						num_ground_trues += 1
						num_false_negative += utt_result['score'][spk_ind]['false'][tar_ind]
					else:
						num_ground_falses += 1
						num_false_positive += utt_result['score'][spk_ind]['false'][tar_ind]


		false_negative_rate = num_false_negative/num_ground_trues
		false_positive_rate = num_false_positive/num_ground_falses

		eer, thr_ind = get_eer(false_negative_rate, false_positive_rate)
		result_summary = {'eer': eer, 'thr': self.thr_range[thr_ind]}
		#

		print ''
		print 'Result for %s (using %s %s): ' % ('eer', self.__class__.__name__, self.scorer_name)
		print 'EER=%.2f%% (threshold=%.3f)' % (result_summary['eer']*100.0, result_summary['thr'])

		if self.make_figure:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			ax.scatter(false_positive_rate*100.0, false_negative_rate*100.0, edgecolors='blue', facecolors='none')
			ax.plot([eer * 100.0], [eer * 100.0], marker='o', markersize=6, color="red")
			ax.annotate('EER=%.1f%% (thr=%.2f)' % (eer*100.0, self.thr_range[thr_ind]), (eer * 100.0, eer * 100.0))
			ax.set_xlim(0.0, 100.0)
			ax.set_ylim(0.0, 100.0)
			ax.set_xlabel('False positive rate (%)')
			ax.set_ylabel('False negative rate (%)')
			fig.savefig(os.path.join(self.store_dir, 'eer_graph_%s.png' % self.scorer_name))
		return result_summary


def get_eer(false_negative_rate, false_positive_rate):

	if false_positive_rate[0] < false_negative_rate[0]:
		raise BaseException('')

	found_eer = False
	for ind in range(len(false_negative_rate)):
		false_neg_ra = false_negative_rate[ind]
		false_pos_ra = false_positive_rate[ind]
		if false_pos_ra < false_neg_ra:
			# crossed the equal error line. We define the EER now as the intersection between the equal error line and
			# the line going through the threshold point right before and right after the crossing of the equal error
			# line.
			x1 = false_negative_rate[ind-1]
			y1 = false_positive_rate[ind-1]
			x2 = false_negative_rate[ind]
			y2 = false_positive_rate[ind]

			eer = ((x2 - x1)*y1 - (y2 - y1)*x1) / ((x2 - x1) - (y2 - y1))
			found_eer = True
			break

	if not found_eer:
		raise BaseException('Did not manage to find an EER.')

	return eer, ind