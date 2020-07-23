"""@file equal_error_rate.py
contains the EER class"""

import os
import numpy as np
import scorer
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class EER(scorer.Scorer):
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

		super(EER, self).__init__(conf, evalconf, dataconf, store_dir, numbatches, task, scorer_name, checkpoint_file)

		thr_range_parse = map(float, conf['thr_range'].split(' '))
		self.thr_range = np.arange(thr_range_parse[0], thr_range_parse[1], thr_range_parse[2])

		self.nr_true_S = int(conf['num_true_speakers'])

		self.target = np.concatenate([np.ones(self.nr_true_S, dtype=np.bool), np.zeros(self.nrS - self.nr_true_S, dtype=np.bool)])

		if 'make_figure' in conf and conf['make_figure'] == 'True':
			self.make_figure = True
		else:
			self.make_figure = False

	def _get_score(self, handled_output, enr_vecs, labels):
		"""score the handled output.

		Args:
			handled_output

		Returns:
			the score"""

		score_to_binary = [[outp > thr for thr in self.thr_range] for outp in handled_output]

		score_dict = list()
		for spk_ind, spk_target in zip(range(self.nrS), labels):
			spk_binaries = score_to_binary[spk_ind]
			score_dict.append(dict())
			score_dict[spk_ind]['binary'] = spk_binaries
			score_dict[spk_ind]['target'] = spk_target

			correct = np.equal(spk_binaries, spk_target)
			score_dict[spk_ind]['false'] = np.logical_not(correct)

		return score_dict

	def get_enr_vecs(self, utt_ind):
		enr_vecs = None
		labels = self.target
		return enr_vecs, labels

	def storable_result(self):
		storable_result = copy.deepcopy(self.results)
		for utt in storable_result:
			for spk_ind in range(len(storable_result[utt]['score'])):
				storable_result[utt]['score'][spk_ind]['binary'] = map(int, storable_result[utt]['score'][spk_ind]['binary'])
				storable_result[utt]['score'][spk_ind]['false'] = map(int, storable_result[utt]['score'][spk_ind]['false'])
				storable_result[utt]['score'][spk_ind]['target'] = int(storable_result[utt]['score'][spk_ind]['target'])

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
				if utt_result['score'][spk_ind]['target']:
					num_ground_trues += 1
					num_false_negative += utt_result['score'][spk_ind]['false']
				else:
					num_ground_falses += 1
					num_false_positive += utt_result['score'][spk_ind]['false']

		false_negative_rate = num_false_negative/num_ground_trues
		false_positive_rate = num_false_positive/num_ground_falses

		eer, thr_ind = get_eer(false_negative_rate, false_positive_rate)
		result_summary = {'eer': eer, 'thr': self.thr_range[thr_ind]}
		#

		print ''
		print 'Result for %s (using %s): ' % ('eer', self.__class__.__name__)
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
			fig.savefig(os.path.join(self.store_dir, 'eer_graph.png'))
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