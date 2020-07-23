"""@file speaker_verification_handler__factory.py
"""

from . import averager, ivector_extractor, attractor_from_embeddings


def factory(speaker_verification_handler):
    """
    gets a reconstructor class

    Args:
        speaker_verification_handler: the speaker_verification_handler type

    Returns:
        a speaker_verification_handler class
    """

    if speaker_verification_handler == 'averager':
        return averager.Averager
    elif speaker_verification_handler == 'ivector_extractor':
        return ivector_extractor.IvectorExtractor
    elif speaker_verification_handler == 'attractor_from_embeddings':
        return attractor_from_embeddings.AttractorFromEmbeddings
    else:
        raise Exception('Undefined speaker verification handler type: %s' % speaker_verification_handler)
