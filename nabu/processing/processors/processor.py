"""@file processor.py
contains the Processor class"""

from abc import ABCMeta, abstractmethod
import numpy as np


class Processor(object):
	"""general Processor class for data processing"""

	__metaclass__ = ABCMeta

	def __init__(self, conf):
		"""Processor constructor

		Args:
			conf: processor configuration as a dictionary of strings
		"""

		self.conf = conf

	@abstractmethod
	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: a string, can be a line of text a pointer to a file etc.

		Returns:
			The processed data"""

	def segment_data(self, data, time_dim=0, winlen_sample=None, winstep_sample=None):
		"""split the data into segments for all desired segment lengths

		Args:
			data: the data to be split in numpy format. [... x time_steps x ...]
			time_dim: the dimension to segment on.
			winlen_sample: optional, can be given as arguments, when
			segment_lengths is given in frames, but data is given in samples
			winstep_sample: optional, can be given as arguments, when
			segment_lengths is given in frames, but data is given in samples

		Returns:
			the segmented data: a list of [... x segment_length x ...]
		"""

		segmented_data = dict()
		N = np.shape(data)[time_dim]

		for seg_length in self.segment_lengths:
			if seg_length == 'full':
				seg_data = [data]
			else:
				seg_len = int(seg_length)  # in frames

				if winlen_sample is None and winstep_sample is None:
					Nseg = int(np.floor(float(N)/float(seg_len)))
					if Nseg < 1:
						seg_data = [np.concatenate((data, np.zeros((seg_len-N, self.dim))), axis=0)]
					else:
						seg_data = []
						for seg_ind in range(Nseg):
							time_indices = range(seg_ind*seg_len, (seg_ind+1)*seg_len)
							seg_data.append(np.take(data, time_indices, axis=time_dim))

				else:
					winoverlap_sample = winlen_sample - winstep_sample
					seg_len_sample = seg_len*winstep_sample + winoverlap_sample
					Nseg = int(np.floor(float(N-winoverlap_sample)/float(seg_len*winstep_sample)))
					if Nseg < 1:
						seg_data = [np.concatenate((data, np.zeros((seg_len_sample-N, self.dim))), axis=0)]
					else:
						seg_data = []
						for seg_ind in range(Nseg):
							start_offset = seg_ind*(seg_len*winstep_sample)
							time_indices = range(start_offset, start_offset+seg_len_sample)
							seg_data.append(np.take(data, time_indices, axis=time_dim))

			segmented_data[seg_length] = seg_data

		return segmented_data

	@abstractmethod
	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""

	def pre_loop(self, dataconf):
		"""allow the processor to access data before looping over all the data

		Args:
			dataconf: config file on the part of the database being processed"""

	def post_loop(self, dataconf):
		"""allow the processor to access data after looping over all the data

		Args:
			dataconf: config file on the part of the database being processed"""