"""@file tfwriter.py
contains the TfWriter class"""

import os
from abc import ABCMeta, abstractmethod
import tensorflow as tf


class TfWriter(object):
    """A class for writing the TF record files"""

    __metaclass__ = ABCMeta

    def __init__(self, datadir, start_ind=0):
        """TfWriter constructor

        Args:
            datadir: the directory where the data will be written
            start_ind: set the starting file number
        """

        if not os.path.exists(datadir):
            # if the directory does not exist create it
            os.makedirs(datadir)

        # store the path to the scp file
        self.scp_file = os.path.join(datadir, 'pointers.scp')

        # set the current file number to start_ind
        self.filenum = start_ind

        # store te path to the write directory
        self.write_dir = os.path.join(datadir, 'data')
        if self.filenum == 0:
            os.makedirs(self.write_dir)

    def write(self, data, name):
        """write data to a file

        Args:
            data: the data to be written
            name: the name of the data"""

        # creater the example
        example = self._get_example(data)

        # the filename for this example
        filename = os.path.join(self.write_dir, 'file%d' % self.filenum)
        self.filenum += 1

        # write the example to file
        writer = tf.python_io.TFRecordWriter(filename)
        writer.write(example.SerializeToString())
        writer.close()

        # put a pointer in the scp file
        with open(self.scp_file, 'a') as fid:
            fid.write('%s\t%s\n' % (name, filename))

    @abstractmethod
    def _get_example(self, data):
        """write data to a file

        Args:
            data: the data to be written"""
