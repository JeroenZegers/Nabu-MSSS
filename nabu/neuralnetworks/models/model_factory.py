'''@file model_factory.py
contains the model factory'''

from . import dblstm, plain_variables, linear, dblstm_linear

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
    elif architecture == 'dblstm_linear':
        return dblstm_linear.DBLSTMLinear
    else:
        raise Exception('undefined architecture type: %s' % architecture)
