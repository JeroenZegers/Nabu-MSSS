'''@file processor_factory.py
contains the Processor factory method'''

from . import audio_feat_processor, onehotperfeature_target_processor, scorelabelperfeature_processor, \
audio_multi_signal_processor, audio_signal_processor, multi_target_processor, strlabel2index_processor, \
matrix2vector_processor, noise_ratio_processor


def factory(processor):
    '''gets a Processor class

    Args:
        processor: the processor type

    Returns:
        a Processor class'''

    if processor == 'audio_feat_processor':
        return audio_feat_processor.AudioFeatProcessor
    elif processor == 'onehotperfeature_target_processor':
        return onehotperfeature_target_processor.onehotperfeatureTargetProcessor
    elif processor == 'multi_target_processor':
        return multi_target_processor.MultiTargetProcessor
    elif processor == 'scorelabelperfeature_processor':
        return scorelabelperfeature_processor.ScorelabelperfeatureProcessor
    elif processor == 'audio_multi_signal_processor':
        return audio_multi_signal_processor.AudioMultiSignalProcessor
    elif processor == 'audio_signal_processor':
        return audio_signal_processor.AudioSignalProcessor
    elif processor == 'strlabel2index_processor':
        return strlabel2index_processor.Strlabel2indexProcessor
    elif processor == 'matrix2vector_processor':
        return matrix2vector_processor.Matrix2VectorProcessor
    elif processor == 'noise_ratio_processor':
        return noise_ratio_processor.noiseRatioProcessor
    else:
        raise Exception('unknown processor type: %s' % processor)
