import os
from nabu.postprocessing import data_reader
import numpy as np

class DummyReconstructor(object):
	requested_output_names = []

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		# Whether the raw output should also be stored (besides the reconstructed audiosignal)
		self.store_output = conf['store_output'] == 'True'
		if self.store_output:
			self.output_dir = os.path.join(rec_dir, 'raw_output')
			if not os.path.isdir(self.output_dir):
				os.makedirs(self.output_dir)

			self.requested_output_names = conf['output_names'].split(' ')

			self.pos = 0
			if evalconf.has_option(task, 'batch_size'):
				self.batch_size = int(evalconf.get(task, 'batch_size'))
			else:
				self.batch_size = int(evalconf.get('evaluator', 'batch_size'))
			self.segment_lengths = evalconf.get('evaluator', 'segment_length').split(' ')

			# get the original mixtures reader
			org_mix_names = conf['org_mix'].split(' ')
			org_mix_dataconfs = []
			for org_mix_name in org_mix_names:
				org_mix_dataconfs.append(dict(dataconf.items(org_mix_name)))
			self.org_mix_reader = data_reader.DataReader(org_mix_dataconfs, self.segment_lengths)

	def __call__(self, batch_outputs, batch_sequence_lengths):
		if self.store_output:
			for utt_ind in range(self.batch_size):
				utt_output = dict()
				for output_name in self.requested_output_names:
					if output_name == 'act_logit':
						utt_output[output_name] = batch_outputs[output_name][utt_ind][:, :batch_sequence_lengths[output_name][utt_ind], :]
					else:
						raise BaseException('Not yet implemented for other outputs.')

				utt_name = self.org_mix_reader.get_name_for_pos(self.pos)

				for output_name in self.requested_output_names:
					savename = output_name+'_'+ utt_name
					np.save(os.path.join(self.output_dir, savename), utt_output[output_name])

				self.pos += 1

	def opt_frame_perm(self, batch_outputs, batch_targets, batch_sequence_lengths):
		pass

	def write_audiofile(self, reconstructed_signals, utt_info):
		pass
