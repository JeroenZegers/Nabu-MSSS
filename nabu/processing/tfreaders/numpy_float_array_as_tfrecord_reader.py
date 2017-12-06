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

        ##read the dimension of the data
        #with open(os.path.join(datadirs[0], 'dim')) as fid:
            #metadata['dim'] = int(fid.read())
        #for datadir in datadirs:
            #with open(os.path.join(datadir, 'dim')) as fid:
                #if metadata['dim'] != int(fid.read()):
                    #raise Exception(
                        #'all audio feature reader dimensions must be the same')

        #read the non-time dimensions of the data

        with open(os.path.join(datadirs, 'nontime_dims')) as fid:
            metadata['nontime_dims'] = fid.read().strip().split(',')
            metadata['nontime_dims'] = map(int,metadata['nontime_dims'])

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
