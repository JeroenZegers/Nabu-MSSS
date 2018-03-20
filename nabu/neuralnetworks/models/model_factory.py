'''@file model_factory.py
contains the model factory'''

from . import dblstm, plain_variables, linear, concat, sigmoid,relu

def factory(architecture):
    '''get a model class

    Args:
        conf: the model conf

    Returns:
        a model class'''

    if architecture == 'dblstm':
        return dblstm.DBLSTM
    elif architecture == 'linear':
        return linear.Linear
    elif architecture == 'plain_variables':
        return plain_variables.PlainVariables
    elif architecture == 'concat':
        return concat.Concat
    elif architecture == 'sigmoid':
        return sigmoid.Sigmoid
    elif architecture == 'RELU':
        return relu.RELU
    else:
        raise Exception('undefined architecture type: %s' % architecture)
