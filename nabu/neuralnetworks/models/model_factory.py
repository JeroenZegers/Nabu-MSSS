"""@file model_factory.py
contains the model factory"""

from . import dblstm, plain_variables, linear, concat, leaky_dblstm, reconstruction_layer, \
  multi_averager, feedforward, leaky_dblstm_iznotrec, leaky_dblstm_notrec, dbrnn,\
  capsnet, dbr_capsnet, dblstm_capsnet, dbgru, leaky_dbgru, dbresetgru, dbresetlstm, dlstm, dresetlstm,\
  leaky_dlstm, encoder_decoder_cnn, regular_cnn, framer, dcnn, ntm, ntt_rec


def factory(architecture):
    """get a model class

    Args:
        conf: the model conf

    Returns:
        a model class"""

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
    elif architecture == 'dblstm_capsnet':
        return dblstm_capsnet.DBLSTMCapsNet
    elif architecture == 'dbgru':
        return dbgru.DBGRU
    elif architecture == 'leaky_dbgru':
        return leaky_dbgru.LeakyDBGRU
    elif architecture == 'dbresetgru':
        return dbresetgru.DBResetGRU
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
    elif architecture == 'dcnn':
        return dcnn.DCNN
    elif architecture == 'regular_cnn':
        return regular_cnn.RegularCNN
    elif architecture == 'framer':
        return framer.Framer
    elif architecture == 'deframer_select':
        return framer.DeframerSelect
    elif architecture == 'reconstruction_layer':
        return reconstruction_layer.Reconstruction_Layer
    elif architecture == 'ntm':
        return ntm.NTM
    elif architecture == 'ntt_rec':
        return ntt_rec.NTTRec
    else:
        raise Exception('undefined architecture type: %s' % architecture)
