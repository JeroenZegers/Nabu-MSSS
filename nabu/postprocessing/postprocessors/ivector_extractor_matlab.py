"""@file ivector_extractor.py
contains the IvectorExtractor class"""
import os
import numpy as np
import postprocessor
from nabu.processing.feature_computers import feature_computer_factory
import matlab.engine
import matlab

class IvectorExtractor(postprocessor.Postprocessor):
	"""the ivector extractor class

	a ivector extractor is used to extract ivectors from (reconstructed) signals"""

	def __init__(self, conf, evalconf,  expdir, rec_dir, postprocessors_name, name=None):
		"""IvectorExtractor constructor

		Args:
			conf: the ivector_extractor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			expdir: the experiment directory
			rec_dir: the directory where the reconstructions are
			task: name of the task
		"""

		super(IvectorExtractor, self).__init__(
			conf, evalconf, expdir, rec_dir, postprocessors_name, name)

		self.lda = conf['lda']=='True'
		if self.lda:
			self.v_dim = conf['v_dim']
		else:
			self.tv_dim = conf['tv_dim']

		self.matlab_eng = matlab.engine.start_matlab("-nodesktop")
		# Go to the directory where the getIvec.m script is
		self.matlab_eng.cd(conf['model_dir'])

	def postproc(self, filenames):
		"""postprocess the signals

		Args:
			output: the signals to be postprocessed

		Returns:
			the post processing data"""

		data = []
		for filename in filenames:
			filename_parts = filename.split('/')
			wavname = filename_parts[-1]
			spk = filename_parts[-2]
			filedir = '/'.join(filename_parts[0:-2])

			try:
				if not self.lda:
					mat_output = self.matlab_eng.getiVecforpython(filedir, spk, wavname, self.tv_dim, nargout=1)
				else:
					mat_output = self.matlab_eng.getiVecLDAforpython(filedir, spk, wavname,  self.v_dim, nargout=1)
			except:
				print ''
				print filedir
				raise

			ivec = [mat_out[0] for mat_out in mat_output]
			data.append(ivec)

		return data
