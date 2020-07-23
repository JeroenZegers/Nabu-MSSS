"""@file data_reader.py
contains a reader class for data"""

from six.moves import configparser
from nabu.processing.processors import processor_factory
import gzip
import os


class DataReader(object):
	"""the data reader class.

	a reader for data. Data is not stored in tensorflow format
	as was done in data.py. Data is returned in numpy format
	and is accessed by indexing instead of looping over all
	data. It is currently only used in postprocessing.
	"""

	def __init__(self, dataconfs, segment_lengths=['full']):
		"""DataReader constructor

		Args:
			dataconfs: the database configuration
			segment_lengths: A list containing the desired lengths of segments.
			Possibly multiple segment lengths
		"""

		if len(segment_lengths) > 1:
			print(
				'Warning: Not yet implemented __call__ correctly for multiple segments. The returned utt_info, does not ' \
				'contain the _part sufix and processed returns only 1 processed')
		self.segment_lengths = segment_lengths

		self.processors = []
		self.start_index_set = [0]
		self.datafile_lines = []
		for dataconf in dataconfs:
			# read the processor config
			proc_cfg_file = dataconf['processor_config']
			if not os.path.isfile(proc_cfg_file):
				raise BaseException('%s does not exist' % proc_cfg_file)
			parsed_proc_cfg = configparser.ConfigParser()
			parsed_proc_cfg.read(proc_cfg_file)
			proc_cfg = dict(parsed_proc_cfg.items('processor'))

			# create a processor
			self.processors.append(processor_factory.factory(proc_cfg['processor'])(proc_cfg, self.segment_lengths))

			# get the datafiles lines
			datafile = dataconf['datafiles']  # TODO: for the moment expecting only 1 file, but this also makes sense?
			if datafile[-3:] == '.gz':
				open_fn = gzip.open
			else:
				open_fn = open
			f = open_fn(datafile)
			datalines = f.readlines()
			self.start_index_set.append(self.start_index_set[-1]+len(datalines))
			self.datafile_lines.extend(datalines)

	def __call__(self, list_pos):
		"""read data from the datafile list

		Args:
			list_pos: position on the datafile list to read

		Returns:
			The processed data as a numpy array"""

		line = self.datafile_lines[list_pos]
		for ind, start_index in enumerate(self.start_index_set):
			if start_index > list_pos:
				processor = self.processors[ind-1]
				break

		# split the name and the data line
		splitline = line.strip().split(' ')
		utt_name = splitline[0]
		dataline = ' '.join(splitline[1:])

		# process the dataline
		processed, utt_info = processor(dataline)
		utt_info['utt_name'] = utt_name

		# Currently only returning 1 processed!
		processed = processed[self.segment_lengths[0]][0]

		return processed, utt_info

	def get_name_for_pos(self, list_pos):
		""" get the name of the utterance for the given position from the datafile list

		Args:
			list_pos: position on the datafile list to read

		Returns:
			The name of the utterance"""

		line = self.datafile_lines[list_pos]

		# split the name and the data line
		splitline = line.strip().split(' ')
		utt_name = splitline[0]

		return utt_name
