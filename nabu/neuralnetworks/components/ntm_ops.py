""" file ntm_ops
Some operations for Neural Turing Machines"""

import numpy as np
import tensorflow as tf


def create_linear_initializer(input_size, dtype=tf.float32):
	stddev = 1.0 / np.sqrt(input_size)
	return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
