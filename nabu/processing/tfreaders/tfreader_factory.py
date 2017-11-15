'''@file tfreader_factory.py
contains the tfreader factory'''

from . import numpy_float_array_as_tfrecord_reader, numpy_bool_array_as_tfrecord_reader,\
  index_list_as_tfrecord_reader

def factory(writer_style):
    '''factory for tfreaders

    Args:
        writer_style: the way the data was written

    Returns:
        a tfreader class
    '''

    if writer_style == 'numpy_float_array_as_tfrecord':
        return numpy_float_array_as_tfrecord_reader.NumpyFloatArrayAsTfrecordReader
    elif writer_style == 'numpy_bool_array_as_tfrecord':
        return numpy_bool_array_as_tfrecord_reader.NumpyBoolArrayAsTfrecordReader
    elif writer_style == 'index_list_as_tfrecord':
        return index_list_as_tfrecord_reader.IndexListAsTfrecordReader
    else:
        raise Exception('unknown writer style: %s' % writer_style)
