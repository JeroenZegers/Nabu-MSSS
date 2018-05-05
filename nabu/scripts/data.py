'''@file data.py
does the data preperation'''

import os
from six.moves import configparser
import gzip
import tensorflow as tf
from nabu.processing.processors import processor_factory
from nabu.processing.tfwriters import tfwriter_factory
import pdb

def main(expdir):
    '''main function'''

    #read the data conf file
    parsed_cfg = configparser.ConfigParser()
    parsed_cfg.read(os.path.join(expdir, 'database.cfg'))

    #loop over the sections in the data config
    name = parsed_cfg.sections()[0]

    #read the section
    conf = dict(parsed_cfg.items(name))

    #the length of the segments. Possibly multiple segment lengths
    if 'segment_lengths' in conf:
	segment_lengths = conf['segment_lengths'].split(' ')
    else:
	segment_lengths = ['full']

    if not os.path.exists(conf['store_dir']):
	os.makedirs(conf['store_dir'])
    else:
	print '%s already exists, skipping this section' % conf['store_dir']
	return

    #read the processor config
    parsed_proc_cfg = configparser.ConfigParser()
    parsed_proc_cfg.read(os.path.join(expdir, 'processor.cfg'))
    proc_cfg = dict(parsed_proc_cfg.items('processor'))

    #create a processor
    processor = processor_factory.factory(proc_cfg['processor'])(proc_cfg, segment_lengths)

    #create the writers
    writers = dict()
    for seg_length in segment_lengths:
	writer_store_dir = os.path.join(conf['store_dir'],seg_length)
	writers[seg_length] = tfwriter_factory.factory(conf['writer_style'])(writer_store_dir)

    #before looping over the data, allow the processor to access the data (e.g.
    #for global mean and variance calculation) (or should this be done in init?)
    processor.pre_loop(conf)

    #loop over the data files
    for datafile in conf['datafiles'].split(' '):
	if datafile[-3:] == '.gz':
	    open_fn = gzip.open
	else:
	    open_fn = open

	#loop over the lines in the datafile
	for line in open_fn(datafile):
	    #split the name and the data line
	    splitline = line.strip().split(' ')
	    utt_name = splitline[0]
	    dataline = ' '.join(splitline[1:])

	    #process the dataline
	    processed, _ = processor(dataline)

	    #write the processed data to disk
	    for seg_length in segment_lengths:

		for i,proc_seg in enumerate(processed[seg_length]):

		    seg_utt_name = utt_name + '_part %d' %i
		    writers[seg_length].write(proc_seg, seg_utt_name)

    #after looping over the data, allow the processor to access the data
    processor.post_loop(conf)

    #write the metadata to file
    processor.write_metadata(conf['store_dir'])


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
    FLAGS = tf.app.flags.FLAGS

    main(FLAGS.expdir)
