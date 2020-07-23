"""@file processor_factory.py
contains the Processor factory method"""

from . import audio_feat_processor, onehotperfeature_target_processor, scorelabelperfeature_processor, \
audio_multi_signal_processor, audio_signal_processor, multi_target_processor, strlabel2index_processor, \
matrix2vector_processor, fracscorelabelperfeature_processor, onehotperfeature_target_dummy_processor,\
multi_target_dummy_processor, scorelabelperfeatureinmixture_processor, audio_feat_conc_processor,\
scorelabelperfeature_multimic_processor, matrix2vector_processor, ideal_ratio_processor, one_processor,\
multi_averager_encapsulator, spatial_feat_processor, onehotperfeature_target_multimic_processor,\
multi_target_multimic_processor, ideal_ratio_multimic_processor, vad_timings_processor, index_processor


def factory(processor):
    """gets a Processor class

    Args:
        processor: the processor type

    Returns:
        a Processor class"""

    if processor == 'audio_feat_processor':
        return audio_feat_processor.AudioFeatProcessor
    elif processor == 'onehotperfeature_target_processor':
        return onehotperfeature_target_processor.onehotperfeatureTargetProcessor
    elif processor == 'onehotperfeature_target_multimic_processor':
        return onehotperfeature_target_multimic_processor.onehotperfeatureTargetMultimicProcessor
    elif processor == 'multi_target_processor':
        return multi_target_processor.MultiTargetProcessor
    elif processor == 'multi_target_multimic_processor':
        return multi_target_multimic_processor.MultiTargetMultimicProcessor
    elif processor == 'scorelabelperfeature_processor':
        return scorelabelperfeature_processor.ScorelabelperfeatureProcessor
    elif processor == 'scorelabelperfeature_multimic_processor':
        return scorelabelperfeature_multimic_processor.ScorelabelperfeatureMultimicProcessor
    elif processor == 'scorelabelperfeatureinmixture_processor':
        return scorelabelperfeatureinmixture_processor.ScorelabelperfeatureinmixtureProcessor
    elif processor == 'fracscorelabelperfeature_processor':
        return fracscorelabelperfeature_processor.FracScorelabelperfeatureProcessor
    elif processor == 'audio_multi_signal_processor':
        return audio_multi_signal_processor.AudioMultiSignalProcessor
    elif processor == 'audio_signal_processor':
        return audio_signal_processor.AudioSignalProcessor
    elif processor == 'strlabel2index_processor':
        return strlabel2index_processor.Strlabel2indexProcessor
    elif processor == 'matrix2vector_processor':
        return matrix2vector_processor.Matrix2VectorProcessor
    elif processor == 'matrix2matrix_processor':
        return matrix2vector_processor.Matrix2MatrixProcessor
    elif processor == 'onehotperfeature_target_dummy_processor':
        return onehotperfeature_target_dummy_processor.onehotperfeatureTargetDummyProcessor
    elif processor == 'multi_target_dummy_processor':
        return multi_target_dummy_processor.MultiTargetDummyProcessor
    elif processor == 'audio_feat_conc_processor':
        return audio_feat_conc_processor.AudioFeatConcProcessor
    elif processor == 'multi_averager_encapsulator':
        return multi_averager_encapsulator.MultiAveragerEncapsulator
    elif processor == 'spatial_feat_processor':
        return spatial_feat_processor.SpatialFeatProcessor
    elif processor == 'ideal_ratio_processor':
        return ideal_ratio_processor.IdealRatioProcessor
    elif processor == 'ideal_ratio_multimic_processor':
        return ideal_ratio_multimic_processor.IdealRatioMultimicProcessor
    elif processor == 'snr_multimic_processor':
        return ideal_ratio_multimic_processor.SnrMultimicProcessor
    elif processor == 'one_processor':
        return one_processor.OneProcessor
    elif processor == 'zero_processor':
        return one_processor.ZeroProcessor
    elif processor == 'vad_timings_processor':
        return vad_timings_processor.VadTimingsProcessor
    elif processor == 'vad_timings_samples_processor':
        return vad_timings_processor.VadTimings2SamplesProcessor
    elif processor == 'index_processor':
        return index_processor.indexProcessor
    else:
        raise Exception('unknown processor type: %s' % processor)
