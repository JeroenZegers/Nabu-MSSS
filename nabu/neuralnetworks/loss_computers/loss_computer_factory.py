'''@file loss_computer_factory.py
contains the Loss computer factory mehod'''

from . import deepclustering_loss, pit_loss, l41_loss, pit_l41_loss,\
  deepclustering_L1_loss, ldajer_loss, dist2mean_rat_loss,\
  dist2mean_rat_squared_loss, intravar2centervar_rat_loss,\
  dist2mean_rat_fracbins_loss, crossentropy_multi_loss,\
  dist2mean_closest_rat_loss,\
  direct_loss, dist2mean_epsilon_closest_rat_loss,\
  dc_pit_loss, crossentropy_multi_loss_reshapelogits,\
  crossentropy_multi_loss_reshapelogits_avtime,\
  deepclustering_full_crossentropy_multi_reshapedlogits_avtime_loss,\
  deepclustering_2and3spk_loss

def factory(loss_type):
    '''gets a Loss computer class

    Args:
        loss_type: the loss type

    Returns: a Loss computer class
    '''

    if loss_type == 'deepclustering':
        return deepclustering_loss.DeepclusteringLoss
    elif loss_type == 'deepclustering_2and3spk':
        return deepclustering_2and3spk_loss.Deepclustering2and3SpkLoss
    elif loss_type == 'pit':
        return pit_loss.PITLoss
    elif loss_type == 'l41':
        return l41_loss.L41Loss
    elif loss_type == 'pit_l41':
        return pit_l41_loss.PITL41Loss
    elif loss_type == 'deepclustering_l1':
        return deepclustering_L1_loss.DeepclusteringL1Loss
    elif loss_type == 'ldajer':
        return ldajer_loss.LdaJerLoss
    elif loss_type == 'intravar2centervar_rat':
        return intravar2centervar_rat_loss.IntraVar2CenterVarRatLoss
    elif loss_type == 'dist2mean_rat':
        return dist2mean_rat_loss.Dist2MeanRatLoss
    elif loss_type == 'dist2mean_rat_squared':
        return dist2mean_rat_squared_loss.Dist2MeanRatSquaredLoss
    elif loss_type == 'dist2mean_rat_fracbins':
        return dist2mean_rat_fracbins_loss.Dist2MeanRatFracBinsLoss
    elif loss_type == 'dist2mean_closest_rat':
        return dist2mean_closest_rat_loss.Dist2MeanClosestRatLoss
    elif loss_type == 'dist2mean_epsilon_closest_rat':
        return dist2mean_epsilon_closest_rat_loss.Dist2MeanEpsilonClosestRatLoss
    elif loss_type == 'direct':
        return direct_loss.DirectLoss
    elif loss_type == 'crossentropy_multi':
        return crossentropy_multi_loss.CrossEntropyMultiLoss
    elif loss_type == 'crossentropy_multi_reshapelogits':
        return crossentropy_multi_loss_reshapelogits.CrossEntropyMultiLossReshapeLogits
    elif loss_type == 'crossentropy_multi_reshapelogits_avtime':
        return crossentropy_multi_loss_reshapelogits_avtime.CrossEntropyMultiLossReshapeLogitsAvTime
    elif loss_type == 'dc_pit':
        return dc_pit_loss.DcPitLoss
    elif loss_type == 'deepclustering_full_crossentropy_multi_reshapelogits_avtime':
        return deepclustering_full_crossentropy_multi_reshapedlogits_avtime_loss.DeepclusteringFullCrossEntropyMultiReshapedLogitsAvTimeLoss
    else:
        raise Exception('Undefined loss type: %s' % loss_type)
