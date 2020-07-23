"""@file data.py
does the data preperation"""

import os
from six.moves import configparser
import gzip
import tensorflow as tf
from nabu.processing.processors import processor_factory
from nabu.processing.tfwriters import tfwriter_factory
import shutil


def main(expdir):
	"""main function"""
	# read the data conf file
	parsed_cfg = configparser.ConfigParser()
	parsed_cfg.read(os.path.join(expdir, 'database.cfg'))

	# loop over the sections in the data config
	name = parsed_cfg.sections()[0]

	# read the section
	conf = dict(parsed_cfg.items(name))
	
	# the length of the segments. Possibly multiple segment lengths
	if 'segment_lengths' in conf:
		segment_lengths = conf['segment_lengths'].split(' ')
	else:
		segment_lengths = ['full']

	if conf['store_dir'] == '/esat/spchtemp/scratch/jzegers/dataforTF/sreMix_segmented_DANet_recs/singlefeatures_hamming_scipy/train_150k':
		start_ind = 106370
		start_ind_per_segment_length = {'500': 239577, 'full': 106370}
		segment_lengths_still_to_process = segment_lengths
	else:
		start_ind = 0
		start_ind_per_segment_length = {seg_len: 0 for seg_len in segment_lengths}

		if not os.path.exists(conf['store_dir']):
			os.makedirs(conf['store_dir'])
			segment_lengths_still_to_process = segment_lengths

			# copy config files to store_dir for archive purposes
			shutil.copyfile(os.path.join(expdir, 'database.cfg'), os.path.join(conf['store_dir'], 'database.cfg'))
			shutil.copyfile(os.path.join(expdir, 'processor.cfg'), os.path.join(conf['store_dir'], 'processor.cfg'))
		else:
			tmp = os.listdir(conf['store_dir'])
			if all([seg_len in tmp for seg_len in segment_lengths]):
				print('%s already exists, skipping this section' % conf['store_dir'])
				return
			else:
				segment_lengths_still_to_process = [seg_len for seg_len in segment_lengths if seg_len not in tmp]
	
	# read the processor config
	parsed_proc_cfg = configparser.ConfigParser()
	parsed_proc_cfg.read(os.path.join(expdir, 'processor.cfg'))
	proc_cfg = dict(parsed_proc_cfg.items('processor'))

	# create a processor
	processor = processor_factory.factory(proc_cfg['processor'])(proc_cfg, segment_lengths_still_to_process)

	# create the writers
	writers = dict()
	for seg_length in segment_lengths_still_to_process:
		writer_store_dir = os.path.join(conf['store_dir'], seg_length)
		writers[seg_length] = tfwriter_factory.factory(conf['writer_style'])(
			writer_store_dir, start_ind=start_ind_per_segment_length[seg_length])

	# before looping over the data, allow the processor to access the data (e.g.
	# for global mean and variance calculation) (or should this be done in init?)
	processor.pre_loop(conf)

	# loop over the data files
	for datafile in conf['datafiles'].split(' '):
		if datafile[-3:] == '.gz':
			open_fn = gzip.open
		else:
			open_fn = open

		# loop over the lines in the datafile
		ind = 0
		for line in open_fn(datafile):
			print(ind)
			if ind < start_ind:
				ind += 1
				continue
			# split the name and the data line
			splitline = line.strip().split(' ')
			utt_name = splitline[0]
			dataline = ' '.join(splitline[1:])

			# process the dataline
			processed, _ = processor(dataline)

			# write the processed data to disk
			for seg_length in segment_lengths_still_to_process:
				for i, proc_seg in enumerate(processed[seg_length]):
					seg_utt_name = utt_name + '_part %d' % i
					writers[seg_length].write(proc_seg, seg_utt_name)
			ind += 1

	# after looping over the data, allow the processor to access the data
	processor.post_loop(conf)

	# write the metadata to file
	processor.write_metadata(conf['store_dir'])


if __name__ == '__main__':
	tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
	FLAGS = tf.app.flags.FLAGS

	main(FLAGS.expdir)
