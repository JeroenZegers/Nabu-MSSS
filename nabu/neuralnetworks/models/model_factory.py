'''@file model_factory.py
contains the model factory'''

from . import dblstm, plain_variables, linear, concat, leaky_dblstm,\
  multi_averager, feedforward, leaky_dblstm_iznotrec, leaky_dblstm_notrec, dbrnn,\
  capsnet, dbr_capsnet, dbgru, leaky_dbgru, dbresetlstm, dlstm, dresetlstm,\
  leaky_dlstm, encoder_decoder_cnn

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
    elif architecture == 'dbrnn':
        return dbrnn.DBRNN
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
    elif architecture == 'capsnet':
        return capsnet.CapsNet
    elif architecture == 'dbr_capsnet':
        return dbr_capsnet.DBRCapsNet
    elif architecture == 'dbgru':
        return dbgru.DBGRU
    elif architecture == 'leaky_dbgru':
        return leaky_dbgru.LeakyDBGRU
    elif architecture == 'dbresetlstm':
        return dbresetlstm.DBResetLSTM
    elif architecture == 'dlstm':
        return dlstm.DLSTM
    elif architecture == 'dresetlstm':
        return dresetlstm.DResetLSTM
    elif architecture == 'leaky_dlstm':
        return leaky_dlstm.LeakyDLSTM
    elif architecture == 'encoder_decoder_cnn':
        return encoder_decoder_cnn.EncoderDecoderCNN
    else:
        raise Exception('undefined architecture type: %s' % architecture)
