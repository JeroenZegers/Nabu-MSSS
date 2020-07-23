import numpy as np
import speaker_verification_handler
import os
import json
from nabu.postprocessing import data_reader


class AttractorFromEmbeddings(speaker_verification_handler.SpeakerVerificationHandler):
	""" """

	base_requested_output_names = ['speaker_bin_embeddings']

	def __init__(self, conf, evalconf, dataconf, store_dir, exp_dir, task):

		super(AttractorFromEmbeddings, self).__init__(conf, evalconf, dataconf, store_dir, exp_dir, task)

		self.cut_to_seq_length = True

		self.emb_dim = int(conf['emb_dim'])

		self.task_for_masks = self.conf['task_for_masks']
		if self.task_for_masks in ['here', 'None', task]:
			# Find the masks here. This only makes sense if the same embeddings were used for the speaker separation and
			# speaker recognition task
			raise NotImplementedError()
		else:
			masks_pointer_file = os.path.join(exp_dir, 'reconstructions', self.task_for_masks, 'masks_pointers.scp')
			self.mix2maskfile = dict()
			with open(masks_pointer_file, 'r') as masks_pointer_fid:
				for line in masks_pointer_fid:
					splitline = line.strip('\n').split(' ')
					self.mix2maskfile[splitline[0]] = splitline[1]

			if 'score_type_for_perm' not in conf:
				raise BaseException('')
			result_file = os.path.join(
				exp_dir, 'results_%s_%s_complete.json' % (self.task_for_masks, conf['score_type_for_perm']))
			with open(result_file, 'r') as result_fid:
				all_results = json.load(result_fid)
			self.mix2permutation = {name: all_results[name]['score']['perm']['SS'] for name in all_results.keys()}

		if 'thr_for_mask' in conf:
			self.thr_for_mask = float(conf['thr_for_mask'])
			self.binary_masks = conf['binary_masks'] == 'True'
			if not self.binary_masks:
				self.rescale_masks = conf['rescale_masks'] == 'True'
		else:
			self.thr_for_mask = False

		usedbins_names = conf['usedbins'].split(' ')
		usedbins_dataconfs = []
		for usedbins_name in usedbins_names:
			usedbins_dataconfs.append(dict(dataconf.items(usedbins_name)))
		self.usedbins_reader = data_reader.DataReader(usedbins_dataconfs, self.segment_lengths)

		if 'normalization' not in self.conf or self.conf['normalization'] == 'True':
			self.normalization = True
		else:
			self.normalization = False

		if 'output_names' in self.conf:
			self.requested_output_names = self.conf['output_names'].split(' ')
		else:
			self.requested_output_names = self.base_requested_output_names
		if len(self.requested_output_names) > 1:
			raise BaseException(
				'Expected the amount of requested output names to be one, but was %d instead' %
				len(self.requested_output_names))

	def handle_output(self, output, utt_name):
		handled_output = output[self.requested_output_names[0]]
		handled_output = np.reshape(handled_output, [np.shape(handled_output)[0], -1, self.emb_dim])

		# get the masks
		masks_file = self.mix2maskfile[utt_name]
		masks = np.load(masks_file)
		usedbins, _ = self.usedbins_reader(self.pos)
		masks = masks * np.expand_dims(usedbins, 0)
		masks = np.expand_dims(masks, -1)
		if self.thr_for_mask:
			binary_masks = masks >= self.thr_for_mask
			if self.binary_masks:
				masks = binary_masks
			else:
				masks = masks * binary_masks
				if self.rescale_masks:
					masks[binary_masks] = masks[binary_masks] - np.min(masks[binary_masks])
					masks[binary_masks] = masks[binary_masks] / np.max(masks[binary_masks])

		# get the attractors
		handled_output = np.expand_dims(handled_output, 0)
		nominator_attr = np.sum(handled_output * masks, axis=(1, 2))
		denominator_attr = np.sum(masks, axis=(1, 2))
		attractors = nominator_attr/denominator_attr

		# get the permutation
		permutation = self.mix2permutation[utt_name]
		attractors = attractors[permutation]

		#
		if self.normalization:
			attractors = attractors/(np.linalg.norm(attractors, axis=-1, keepdims=True)+1e-10)
		return attractors
