'''@file index_list_as_tfrecord_writer.py
contains the IndexListAsTfrecordWriter class'''

import numpy as np
import tensorflow as tf
import tfwriter

class IndexListAsTfrecordWriter(tfwriter.TfWriter):
    '''a TfWriter to write index lists'''

    def _get_example(self, data):
        '''write data to a file

        Args:
            data: the data to be written'''

        shape_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[np.array(np.array(data).astype(np.int32).shape).tostring()]))
        data_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[np.array(data).astype(np.int32).tostring()]))
	

        #create the example proto
        example = tf.train.Example(features=tf.train.Features(feature={
            'shape': shape_feature,
            'data': data_feature}))

        return example
