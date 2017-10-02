'''@file tfwriter_factory
contains the tfwriter factory'''

from . import numpy_float_array_as_tfrecord_writer, numpy_bool_array_as_tfrecord_writer

def factory(writer_style):
    '''
    Args:
        writer_style: the way the data should be written

    Returns:
        a tfwriter class
    '''

    if writer_style == 'numpy_float_array_as_tfrecord':
        return numpy_float_array_as_tfrecord_writer.NumpyFloatArrayAsTfrecordWriter
    elif writer_style == 'numpy_bool_array_as_tfrecord':
        return numpy_bool_array_as_tfrecord_writer.NumpyBoolArrayAsTfrecordWriter
    else:
        raise Exception('unknown writer style: %s' % writer_style)
