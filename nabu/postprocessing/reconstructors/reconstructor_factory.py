"""@file reconstructor_factory.py
contains the Reconstructor factory"""

from . import deepclustering_reconstructor, stackedmasks_reconstructor, deepXclustering_reconstructor, \
    deepattractornet_reconstructor, pit_l41_reconstructor, deepclusteringnoise_reconstructor, \
    deepattractornetnoise_hard_reconstructor, deepattractornetnoise_soft_reconstructor, oracle_reconstructor_noise, \
    deepattractornet_softmax_reconstructor, noisefilter_reconstructor, deepattractornet_noisefilter_reconstructor, \
    parallel_deepclustering_reconstructor, stackedmasks_noise_reconstructor, \
    anchor_deepattractornet_softmax_reconstructor, time_anchor_deepattractornet_softmax_reconstructor, \
    time_anchor_read_heads_deepattractornet_softmax_reconstructor, weighted_anchor_deepattractornet_softmax_reconstructor, \
    dummy_reconstructor, oraclemask_reconstructor


def factory(reconstructor):
    """
    gets a reconstructor class

    Args:
        reconstructor: the reconstructor type

    Returns:
        a reconstructor class
    """

    if reconstructor == 'deepclustering' or reconstructor == 'deepclustering_flat':
        return deepclustering_reconstructor.DeepclusteringReconstructor
    elif reconstructor == 'deepXclustering':
        return deepXclustering_reconstructor.DeepXclusteringReconstructor
    elif reconstructor == 'deepattractornet':
        return deepattractornet_reconstructor.DeepattractorReconstructor
    elif reconstructor == 'deepattractornet_softmax':
        return deepattractornet_softmax_reconstructor.DeepattractorSoftmaxReconstructor
    elif reconstructor == 'anchor_deepattractornet_softmax':
        return anchor_deepattractornet_softmax_reconstructor.AnchorDeepattractorSoftmaxReconstructor
    elif reconstructor == 'time_anchor_deepattractornet_softmax':
        return time_anchor_deepattractornet_softmax_reconstructor.TimeAnchorDeepattractorSoftmaxReconstructor
    elif reconstructor == 'time_anchor_scalar_deepattractornet_softmax':
        return time_anchor_deepattractornet_softmax_reconstructor.TimeAnchorScalarDeepattractorSoftmaxReconstructor
    elif reconstructor == 'time_anchor_spk_weights_deepattractornet_softmax':
        return time_anchor_deepattractornet_softmax_reconstructor.TimeAnchorSpkWeightsDeepattractorSoftmaxReconstructor
    elif reconstructor == 'time_anchor_read_heads_deepattractornet_softmax':
        return time_anchor_read_heads_deepattractornet_softmax_reconstructor.TimeAnchorReadHeadsDeepattractorSoftmaxReconstructor
    elif reconstructor == 'weighted_anchor_deepattractornet_softmax':
        return weighted_anchor_deepattractornet_softmax_reconstructor.WeightedAnchorDeepattractorSoftmaxReconstructor
    elif reconstructor == 'stackedmasks':
        return stackedmasks_reconstructor.StackedmasksReconstructor
    elif reconstructor == 'pit_l41':
        return pit_l41_reconstructor.PITL41Reconstructor
    elif reconstructor == 'parallel_deepclustering':
        return parallel_deepclustering_reconstructor.ParallelDeepclusteringReconstructor
    elif reconstructor == 'deepclusteringnoise':
        return deepclusteringnoise_reconstructor.DeepclusteringnoiseReconstructor
    elif reconstructor == 'deepattractornetnoisehard':
        return deepattractornetnoise_hard_reconstructor.DeepattractornoisehardReconstructor
    elif reconstructor == 'deepattractornetnoisesoft':
        return deepattractornetnoise_soft_reconstructor.DeepattractornoisesoftReconstructor
    elif reconstructor == 'oraclemask':
        return oraclemask_reconstructor.OracleMaskReconstructor
    elif reconstructor == 'oraclenoise':
        return oracle_reconstructor_noise.OracleReconstructor
    elif reconstructor == 'noisefilter':
        return noisefilter_reconstructor.NoiseFilterReconstructor
    elif reconstructor == 'deepattractornet_noisefilter':
        return deepattractornet_noisefilter_reconstructor.DeepattractornoisefilterReconstructor
    elif reconstructor == 'stackedmasks_noise':
        return stackedmasks_noise_reconstructor.StackedmasksNoiseReconstructor
    elif reconstructor == 'dummy':
        return dummy_reconstructor.DummyReconstructor
    else:
        raise Exception('Undefined reconstructor type: %s' % reconstructor)
