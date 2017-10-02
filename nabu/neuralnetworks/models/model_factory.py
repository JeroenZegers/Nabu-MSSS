'''@file model_factory.py
contains the model factory'''

from . import dblstm

def factory(architecture):
    '''get a model class

    Args:
        conf: the model conf

    Returns:
        a model class'''

    if architecture == 'dblstm':
        return dblstm.DBLSTM
    else:
        raise Exception('undefined architecture type: %s' % architecture)
