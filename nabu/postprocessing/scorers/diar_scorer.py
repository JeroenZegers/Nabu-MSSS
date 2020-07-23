"""@file diar_scorer.py
contains the scorer using DER (diarization error rate)"""

import scorer
import numpy as np
import bss_eval
import itertools
import math


class DiarFromSigEstScorer(scorer.Scorer):
	"""the DER scorer class using signal estimates. I believe this is a combination DER and SCER as mentioned in
		https://arxiv.org/pdf/1902.07881.pdf
	"""

	score_metrics = ('DER', 'perm')
	score_metrics_to_summarize = (['DER'])
	score_scenarios = (['SS'])
	score_expects = 'data'
	target_name = 'diar_targets'

	def __init__(self, conf, evalconf, dataconf, rec_dir, numbatches, task, scorer_name, checkpoint_file):
		"""SdrScorer constructor

		Args:
			conf: the scorer configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions are
			numbatches: the number of batches to process
		"""

		# the percentage of frame energy required for a speaker to be considered active.
		self.frame_len = int(conf['frame_len'])
		self.thr_std = float(conf['thr_std'])
		self.weight_frames = conf['weight_frames'] in ['true', 'True']

		super(DiarFromSigEstScorer, self).__init__(conf, evalconf, dataconf, rec_dir, numbatches, task, scorer_name, checkpoint_file)

	def _get_score(self, targets, base_signals, rec_src_signals, utt_rate=None):
		"""

		Args:
			targets: the target vads, as a list of numpy arrarys
			base_signals: the duplicated base signal (original mixture), as a list of numpy arrarys
			rec_src_signals: the reconstructed source signals, as a list of numpy arrarys

		Returns:
			the score"""

		# convert to numpy arrays
		base_signal = base_signals[0][:, 0]
		base_signal_framed = np.convolve(np.abs(base_signal), np.ones(self.frame_len), mode='same')
		num_samples = len(base_signal)
		num_frames = len(base_signal_framed)
		mean = np.mean(base_signal_framed)
		std = np.std(base_signal_framed)

		thr = self.thr_std * std

		#
		rec_src_signals_framed = np.array([np.convolve(np.abs(rec_sr_sig), np.ones(self.frame_len), mode='same') for rec_sr_sig in rec_src_signals])
		vad_est_signals = np.array(rec_src_signals_framed > thr, dtype=np.int)

		if self.weight_frames:
			bin_weights = base_signal_framed/np.sum(base_signal_framed)
		else:
			bin_weights = np.ones(num_frames)/num_frames

		targets = np.transpose(targets[:num_frames])
		nr_spk = np.shape(targets)[0]
		permutations = list(itertools.permutations(range(nr_spk), nr_spk))

		perm_cost = []
		for perm in permutations:
			all_cost = np.sum(np.abs(vad_est_signals[np.array(perm)] - targets), axis=0)
			per_frame_cost = np.min([all_cost, np.ones(np.shape(all_cost))], 0)  # frame can only be wrong or right
			weighted_per_frame_cost = bin_weights * per_frame_cost
			perm_cost.append(np.sum(weighted_per_frame_cost))
		best_perm_ind = np.argmin(perm_cost)
		best_perm = permutations[best_perm_ind]
		best_cost = perm_cost[best_perm_ind]  # cost already normalized via bin_weights

		score_dict = {'DER': {'SS': best_cost}, 'perm': {'SS': best_perm}}

		return score_dict


def framesig_padded(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x, ))):
	"""
	Frame a signal into overlapping frames. Also pad at the start

	Args:
		sig: the audio signal to frame.
		frame_len: length of each frame measured in samples.
		frame_step: number of samples after the start of the previous frame that
			the next frame should begin.
		winfunc: the analysis window to apply to each frame. By default no
			window function is applied.

	Returns:
		an array of frames. Size is NUMFRAMES by frame_len.
	"""

	slen = len(sig)
	if slen <= frame_len:
		numframes = 1
	else:
		numframes = int(math.ceil((1.0*slen)/frame_step))

	padsignal = np.concatenate((np.zeros(frame_len/2 - 1), sig, np.zeros(frame_len/2)))

	indices = (np.tile(np.arange(0, frame_len), (numframes, 1)) +
			   np.tile(np.arange(0, numframes*frame_step, frame_step), (frame_len, 1)).T)
	indices = np.array(indices, dtype=np.int32)
	frames = padsignal[indices]
	win = np.tile(winfunc(frame_len), (numframes, 1))
	return frames*win