import numpy as np
import speaker_verification_handler


class Averager(speaker_verification_handler.SpeakerVerificationHandler):
	""" """

	base_requested_output_names = ['act_logit']

	def __init__(self, conf, evalconf, dataconf, store_dir, exp_dir, task):

		super(Averager, self).__init__(conf, evalconf, dataconf, store_dir, exp_dir, task)

		if 'squeeze' in self.conf and self.conf['squeeze'] == 'True':
			self.squeeze = True
		else:
			self.squeeze = False

		if self.conf['activation'] == 'sigmoid':
			self.activation = 'sigmoid'
		else:
			raise BaseException('Other activations not yet implemented')

		if 'average_dimension' in self.conf:
			self.average_dimension = int(self.conf['average_dimension'])
		else:
			self.average_dimension = -1

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

		if self.squeeze:
			handled_output = np.squeeze(handled_output)
		if self.activation == 'sigmoid':
			handled_output = sigmoid(handled_output)

		handled_output = np.mean(handled_output, axis=self.average_dimension)

		return handled_output


def sigmoid(inp):
	outp = 1 / (1 + np.exp(-inp))
	return outp
