"""@file loss_computer_factory.py
contains the Loss computer factory mehod"""

from . import deepclustering_loss, pit_loss, \
	l41_loss, pit_l41_loss,\
	deepclustering_L1_loss, dist2mean_rat_loss,\
	dist2mean_rat_squared_loss, intravar2centervar_rat_loss,\
	dist2mean_rat_fracbins_loss, crossentropy_multi_loss,\
	dist2mean_closest_rat_loss,direct_loss, dist2mean_epsilon_closest_rat_loss,\
	dc_pit_loss, crossentropy_multi_loss_reshapelogits,\
	crossentropy_multi_loss_reshapelogits_avtime,\
	deepclustering_full_crossentropy_multi_reshapedlogits_avtime_loss,\
	deepclustering_2and3spk_loss, deepclustering_flat_loss, deepclusteringnoise_loss, \
	deepattractornetnoise_hard_loss, deepattractornetnoise_hard_loss, deepattractornetnoise_soft_loss, \
	deepattractornet_loss, noisefilter_loss, deepattractornet_noisefilter_loss, pit_noise_loss, dummy_loss, \
    anchor_deepattractornet_softmax_loss, deepclustering_diar_loss


def factory(loss_type):
    """gets a Loss computer class

    Args:
        loss_type: the loss type

    Returns: a Loss computer class
    """

    if loss_type == 'deepclustering':
        return deepclustering_loss.DeepclusteringLoss
    elif loss_type == 'deepclustering_flat':
        return deepclustering_flat_loss.DeepclusteringFlatLoss
    elif loss_type == 'deepclustering_2and3spk':
        return deepclustering_2and3spk_loss.Deepclustering2and3SpkLoss
    elif loss_type == 'pit':
        return pit_loss.PITLoss
    elif loss_type == 'pit_overspeakerized':
        return pit_loss.PITLossOverspeakerized
    elif loss_type == 'pit_sigmoid':
        return pit_loss.PITLossSigmoid
    elif loss_type == 'pit_sigmoid_scaled':
        return pit_loss.PITLossSigmoidScaled
    elif loss_type == 'deepattractornet_sigmoid':
        return deepattractornet_loss.DeepattractornetSigmoidLoss
    elif loss_type == 'deepattractornet_softmax':
        return deepattractornet_loss.DeepattractornetSoftmaxLoss
    elif loss_type == 'anchor_deepattractornet_softmax':
        return anchor_deepattractornet_softmax_loss.AnchorDeepattractornetSoftmaxLoss
    elif loss_type == 'anchor_normdeepattractornet_softmax':
        return anchor_deepattractornet_softmax_loss.AnchorNormDeepattractornetSoftmaxLoss
    elif loss_type == 'weighted_anchor_normdeepattractornet_softmax':
        return anchor_deepattractornet_softmax_loss.WeightedAnchorNormDeepattractornetSoftmaxLoss
    elif loss_type == 'time_anchor_deepattractornet_softmax':
        return anchor_deepattractornet_softmax_loss.TimeAnchorDeepattractornetSoftmaxLoss
    elif loss_type == 'time_anchor_normdeepattractornet_softmax':
        return anchor_deepattractornet_softmax_loss.TimeAnchorNormDeepattractornetSoftmaxLoss
    elif loss_type == 'time_anchor_read_heads_normdeepattractornet_softmax':
        return anchor_deepattractornet_softmax_loss.TimeAnchorReadHeadsNormDeepattractornetSoftmaxLoss
    elif loss_type == 'time_anchor_read_heads_normdeepattractornet_softmax_framebased':
        return anchor_deepattractornet_softmax_loss.TimeAnchorReadHeadsNormDeepattractornetSoftmaxFramebasedLoss
    elif loss_type == 'l41':
        return l41_loss.L41Loss
    elif loss_type == 'pit_l41':
        return pit_l41_loss.PITL41Loss
    elif loss_type == 'deepclustering_l1':
        return deepclustering_L1_loss.DeepclusteringL1Loss
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
    elif loss_type == 'deepclusteringnoise':
        return deepclusteringnoise_loss.DeepclusteringnoiseLoss
    elif loss_type == 'deepattractornetnoisehard':
        return deepattractornetnoise_hard_loss.DeepattractornetnoisehardLoss
    elif loss_type == 'deepattractornetnoisesoft':
        return deepattractornetnoise_soft_loss.DeepattractornetnoisesoftLoss
    elif loss_type == 'noisefilter':
        return noisefilter_loss.NoisefilterLoss
    elif loss_type == 'deepattractornet_noisefilter':
        return deepattractornet_noisefilter_loss.DeepattractornetnoisefilterLoss
    elif loss_type == 'pit_noise':
        return pit_noise_loss.PITNoiseLoss
    elif loss_type == 'deepclusteringnoise_rat_as_weight':
        return deepclusteringnoise_loss.DeepclusteringnoiseRatAsWeightLoss
    elif loss_type == 'deepclusteringnoise_alpha_as_weight':
        return deepclusteringnoise_loss.DeepclusteringnoiseAlphaAsWeightLoss
    elif loss_type == 'deepclusteringnoise_snr_target':
        return deepclusteringnoise_loss.DeepclusteringnoiseSnrTargetLoss
    elif loss_type == 'deepclusteringnoise_dconly':
        return deepclusteringnoise_loss.DeepclusteringnoiseDConlyLoss
    elif loss_type == 'deepclusteringnoise_noiseonly':
        return deepclusteringnoise_loss.DeepclusteringnoiseNoiseonlyLoss
    elif loss_type == 'dummy_loss':
        return dummy_loss.DummyLoss
    elif loss_type == 'deepclustering_diar':
        return deepclustering_diar_loss.DeepclusteringDiarLoss
    else:
        raise Exception('Undefined loss type: %s' % loss_type)
