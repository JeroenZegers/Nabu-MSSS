'''@file index_list_as_tfrecord_reader.py
contains the IndexListAsTfrecordReader class'''

import os
import numpy as np
import tensorflow as tf
import tfreader
import pdb

class IndexListAsTfrecordReader(tfreader.TfReader):
    '''reader for list of indices'''

    def _read_metadata(self, datadirs):
        '''read the input dimension

            Args:
                datadir: the directory where the metadata was written

            Returns:
                the metadata as a dictionary
        '''
	metadata=dict()            
	
        with open(os.path.join(datadirs[0], 'nrS')) as fid:
            metadata['nrS'] = int(fid.read().strip())
        for datadir in datadirs:
            with open(os.path.join(datadir, 'nrS')) as fid:
                if metadata['nrS'] != int(fid.read().strip()):
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

        data = tf.decode_raw(features['data'], tf.int32)
        resh_dims = [self.metadata['nrS']]
        data = tf.reshape(data, resh_dims)
        sequence_length = tf.constant([1])

        return data, sequence_length
