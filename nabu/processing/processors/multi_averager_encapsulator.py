"""@file multi_averager_encapsulator.py
contains the MultiAveragerEncapsulator class"""

import numpy as np
import processor
import processor_factory


class MultiAveragerEncapsulator(processor.Processor):
	"""Encapsulates a different processor. This encapsulated processor is fed single channel data from multi channel
	data. The different processed data are then averaged"""

	def __init__(self, conf, segment_lengths):
		"""ZeroProcessor constructor

		Args:
			conf: ScorelabelperfeatureProcessor configuration as a dict of strings
			segment_lengths: A list containing the desired lengths of segments.
			Possibly multiple segment lengths"""

		self.encapsulated_processor = processor_factory.factory(conf['encapsulated_processor'])(conf, segment_lengths)
		self.comp = self.encapsulated_processor.comp
		self.n_channels = int(conf['n_channels'])
		super(MultiAveragerEncapsulator, self).__init__(conf)

	def __call__(self, dataline):
		"""process the data in dataline
		Args:
			dataline: either a path to a wav file or a command to read and pipe
				an audio file

		Returns:
			segmented_data: The segmented zeros
			utt_info: some info on the utterance"""

		splitdatalines = dataline.strip().split(' ')

		if 'mvn_type' in self.conf and self.conf['mvn_type'] == 'from_files':
			if 'mvn_from_files' not in splitdatalines:
				raise Exception('Expected files to determine mean and variance')
			else:
				keyword_ind = [ind for ind, splitword in enumerate(splitdatalines) if splitword == 'mvn_from_files'][0]
				mvn_files = splitdatalines[keyword_ind+1:]
				splitdatalines = splitdatalines[:keyword_ind]

				all_mvn_data = []
				for mvn_file in mvn_files:
					mvn_data, utt_info = self.encapsulated_processor(mvn_file)
					all_mvn_data.append(mvn_data)
				combined_mvn_data = np.mean([data['full'][0] for data in all_mvn_data], 0)
				mean_to_use = np.mean(combined_mvn_data, 0, keepdims=True)
				std_to_use = np.std(combined_mvn_data, 0, keepdims=True)

		if len(splitdatalines) > self.n_channels:
			# some encapsulated processors require multichannel data them self
			tmp = []
			for ch_ind in range(self.n_channels):
				to_combine = range(ch_ind, len(splitdatalines), self.n_channels)
				tmp.append(' '.join([splitdatalines[ind] for ind in to_combine]))
			splitdatalines = tmp

		elif len(splitdatalines) < self.n_channels:
			raise BaseException('Received less data lines then requested number of input channels')

		multi_processed = list()
		utt_infos = list()
		for splitdataline in splitdatalines:
			segmented_data, utt_info = self.encapsulated_processor(splitdataline)
			multi_processed.append(segmented_data)
			utt_infos.append(utt_info)

		# average over the segments
		multi_signal_averaged = dict()
		for seg_len in self.encapsulated_processor.segment_lengths:
			multi_signal_seg = [mul_sig[seg_len] for mul_sig in multi_processed]
			n_seg = len(multi_signal_seg[0])
			multi_signal_segs_average = []
			for seg_ind in range(n_seg):
				multi_signal_seg_channels = [multi_signal_seg_ch[seg_ind] for multi_signal_seg_ch in multi_signal_seg]
				multi_signal_seg_average = np.mean(np.array(multi_signal_seg_channels), 0)
				if 'mvn_type' in self.conf and self.conf['mvn_type'] == 'from_files':
					multi_signal_seg_average = (multi_signal_seg_average - mean_to_use) / (std_to_use+1e-12)
				multi_signal_segs_average.append(multi_signal_seg_average)
			multi_signal_averaged[seg_len] = multi_signal_segs_average

		# for the moment just taking utt_info from the first channel
		return multi_signal_averaged, utt_infos[0]

	def write_metadata(self, datadir):
		"""write the processor metadata to disk

		Args:
			datadir: the directory where the metadata should be written"""

		self.encapsulated_processor.write_metadata(datadir)


