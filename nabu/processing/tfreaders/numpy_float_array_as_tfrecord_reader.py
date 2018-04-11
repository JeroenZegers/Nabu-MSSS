'''@file numpy_float_array_as_tfrecord_reader.py
contains the NumpyFloatArrayAsTfrecordReader class'''

import os
import numpy as np
import tensorflow as tf
import tfreader
import pdb

class NumpyFloatArrayAsTfrecordReader(tfreader.TfReader):
    '''reader for numpy float arrays'''

    def _read_metadata(self, datadirs):
        '''read the input dimension

            Args:
                datadir: the directory where the metadata was written

            Returns:
                the metadata as a dictionary
        '''

        metadata = dict()

        #read the non-time dimensions of the data
        with open(os.path.join(datadirs[0], 'nontime_dims')) as fid:
            metadata['nontime_dims'] = fid.read().strip().split(',')
            metadata['nontime_dims'] = map(int,metadata['nontime_dims'])
        for datadir in datadirs:
            with open(os.path.join(datadir, 'nontime_dims')) as fid:
		nontime_dims=fid.read().strip().split(',')
		nontime_dims=map(int,nontime_dims)
                if metadata['nontime_dims'] != nontime_dims:
                    raise Exception(
                        'all reader dimensions must be the same')

        return metadata

    def _create_features(self):
        '''
            creates the information about the features

            Returns:
                A dict mapping feature keys to FixedLenFeature, VarLenFeature,
                and SparseFeature values
        '''

        return {'data': tf.FixedLenFeature([], dtype=tf.string)}

    def _process_features(self, features):
        '''process the read features

        features:
            A dict mapping feature keys to Tensor and SparseTensor values

        Returns:
            a pair of tensor and sequence length
        '''

        data = tf.decode_raw(features['data'], tf.float32)
        resh_dims = [-1] + self.metadata['nontime_dims']
        data = tf.reshape(data, resh_dims)
        sequence_length = tf.shape(data)[0]

        return data, sequence_length
