'''@file model_factory.py
contains the model factory'''

from . import dblstm, plain_variables, linear, concat, leaky_dblstm,\
  multi_averager, feedforward, leaky_dblstm_iznotrec, leaky_dblstm_notrec

def factory(architecture):
    '''get a model class

    Args:
        conf: the model conf

    Returns:
        a model class'''

    if architecture == 'dblstm':
        return dblstm.DBLSTM
    elif architecture == 'leaky_dblstm':
        return leaky_dblstm.LeakyDBLSTM
    elif architecture == 'leaky_dblstm_iznotrec':
        return leaky_dblstm_iznotrec.LeakyDBLSTMIZNotRec
    elif architecture == 'leaky_dblstm_notrec':
        return leaky_dblstm_notrec.LeakyDBLSTMNotRec
    elif architecture == 'linear':
        return linear.Linear
    elif architecture == 'feedforward':
        return feedforward.Feedforward
    elif architecture == 'plain_variables':
        return plain_variables.PlainVariables
    elif architecture == 'concat':
        return concat.Concat
    elif architecture == 'multiaverage':
        return multi_averager.MultiAverager
    else:
        raise Exception('undefined architecture type: %s' % architecture)
