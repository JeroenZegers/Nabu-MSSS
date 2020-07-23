"""@file scorer_factory.py
contains the Scorer factory"""

from . import sdr_scorer, sdr_snr_scorer, sdr_snr_noise_scorer, diar_scorer
# from . import sdr_scorer, sdr_snr_scorer, sdr_snr_noise_scorer, pesq_scorer, diar_scorer, si_sdr_scorer


def factory(scorer):
    """
    gets a scorer class

    Args:
        scorer: the scorer type

    Returns:
        a scorer class
    """

    if scorer == 'sdr':
        return sdr_scorer.SdrScorer
    # elif scorer == 'si_sdr':
    #     return si_sdr_scorer.SiSdrScorer
    elif scorer == 'sdr_snr':
        return sdr_snr_scorer.SdrSnrScorer
    elif scorer == 'sdr_snr_noise':
        return sdr_snr_noise_scorer.SdrSnrNoiseScorer
    # elif scorer == 'pesq':
    #     return pesq_scorer.PESQScorer
    elif scorer == 'diar_from_sig_est':
        return diar_scorer.DiarFromSigEstScorer
    else:
        raise Exception('Undefined scorer type: %s' % scorer)
