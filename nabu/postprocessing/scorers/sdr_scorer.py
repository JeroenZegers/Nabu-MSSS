"""@file sdr_scorer.py
contains the scorer using SDR"""

import scorer
import numpy as np
import bss_eval


class SdrScorer(scorer.Scorer):
	"""the SDR scorer class. Uses the script from
	C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis,
	'mir_eval: A Transparent Implementation of Common MIR Metrics', Proceedings of the 15th
	International Conference on Music Information Retrieval, 2014

	a scorer using SDR"""

	score_metrics = ('SDR', 'SIR', 'SAR', 'perm')
	score_metrics_to_summarize = ('SDR', 'SIR', 'SAR')
	score_scenarios = ('SS', 'base')
	score_expects = 'data'

	def __init__(self, conf, evalconf, dataconf, rec_dir, numbatches, task):
		"""SdrScorer constructor

		Args:
			conf: the scorer configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions are
			numbatches: the number of batches to process
		"""

		super(SdrScorer, self).__init__(conf, evalconf, dataconf, rec_dir, numbatches, task)

	def _get_score(self, org_src_signals, base_signals, rec_src_signals):
		"""score the reconstructed utterances with respect to the original source signals

		Args:
			org_src_signals: the original source signals, as a list of numpy arrarys
			base_signals: the duplicated base signal (original mixture), as a list of numpy arrarys
			rec_src_signals: the reconstructed source signals, as a list of numpy arrarys

		Returns:
			the score"""

		# convert to numpy arrays
		org_src_signals = org_src_signals[:, :, 0]
		base_signals = np.array(base_signals)[:, :, 0]
		rec_src_signals = np.array(rec_src_signals)

		#
		collect_outputs = dict()
		collect_outputs[self.score_scenarios[1]] = bss_eval.bss_eval_sources(org_src_signals, base_signals)
		collect_outputs[self.score_scenarios[0]] = bss_eval.bss_eval_sources(org_src_signals, rec_src_signals)

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
