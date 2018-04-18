'''@file scorer_factory.py
contains the Scorer factory'''

from . import  sdr_scorer, sdr_snr_scorer, pesq_scorer

def factory(scorer):
    '''
    gets a scorer class

    Args:
        scorer: the scorer type

    Returns:
        a scorer class
    '''

    if scorer == 'sdr':
        return sdr_scorer.SdrScorer
    elif scorer == 'sdr_snr':
        return sdr_snr_scorer.SdrSnrScorer
    elif scorer == 'sdr_snr_noise':
        return sdr_snr_noise_scorer.SdrSnrNoiseScorer
    elif scorer == 'pesq':
        return pesq_scorer.PESQScorer
    else:
        raise Exception('Undefined scorer type: %s' % scorer)
