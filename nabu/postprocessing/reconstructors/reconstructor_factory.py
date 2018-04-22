'''@file reconstructor_factory.py
contains the Reconstructor factory'''

from . import  deepclustering_reconstructor, stackedmasks_reconstructor, \
deepXclustering_reconstructor, deepattractornet_reconstructor, pit_l41_reconstructor, \
deepclusteringnoise_reconstructor, deepattractornetnoise_hard_reconstructor, \
deepattractornetnoise_soft_reconstructor, oracle_reconstructor_noise, \
deepattractornet_softmax_reconstructor,noisefilter_reconstructor, \
deepattractornet_noisefilter_reconstructor



def factory(reconstructor):
    '''
    gets a reconstructor class

    Args:
        reconstructor: the reconstructor type

    Returns:
        a reconstructor class
    '''

    if reconstructor == 'deepclustering':
        return deepclustering_reconstructor.DeepclusteringReconstructor
    elif reconstructor == 'deepattractornet':
        return deepattractornet_reconstructor.DeepattractorReconstructor
    elif reconstructor == 'deepattractornet_softmax':
        return deepattractornet_softmax_reconstructor.DeepattractorSoftmaxReconstructor
    elif reconstructor == 'deepXclustering':
        return deepXclustering_reconstructor.DeepXclusteringReconstructor
    elif reconstructor == 'stackedmasks':
        return stackedmasks_reconstructor.StackedmasksReconstructor
    elif reconstructor == 'pit_l41':
        return pit_l41_reconstructor.PITL41Reconstructor
    elif reconstructor == 'deepclusteringnoise':
        return deepclusteringnoise_reconstructor.DeepclusteringnoiseReconstructor
    elif reconstructor == 'deepattractornetnoisehard':
        return deepattractornetnoise_hard_reconstructor.DeepattractornoisehardReconstructor
    elif reconstructor == 'deepattractornetnoisesoft':
        return deepattractornetnoise_soft_reconstructor.DeepattractornoisesoftReconstructor
    elif reconstuctor == 'oraclenoise':
        return oracle_reconstructor_noise.OracleReconstructor
    elif reconstructor == 'noisefilter':
        return noisefilter_reconstructor.NoiseFilterReconstructor
    elif reconstructor == 'deepattractornet_noisefilter':
        return deepattractornet_noisefilter_reconstructor.DeepattractornoisefilterReconstructor
    else:
        raise Exception('Undefined reconstructor type: %s' % reconstructor)
