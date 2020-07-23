import numpy as np
import speaker_verification_handler


class IvectorExtractor(speaker_verification_handler.SpeakerVerificationHandler):
	""" """

	base_requested_output_names = ['attractor']

	def __init__(self, conf, evalconf, dataconf, store_dir, exp_dir, task):

		super(IvectorExtractor, self).__init__(conf, evalconf, dataconf, store_dir, exp_dir, task)

		self.cut_to_seq_length = False

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
				'Expected the amount of requested output names to be one, but was %d isntead' %
				len(self.requested_output_names))

	def handle_output(self, output, utt_name):
		handled_output = output[self.requested_output_names[0]]

		if self.normalization:
			handled_output = handled_output/(np.linalg.norm(handled_output, axis=-1, keepdims=True)+1e-10)
		return handled_output
