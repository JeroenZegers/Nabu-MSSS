"""@file scorer_factory.py
contains the Scorer factory"""

from . import equal_error_rate, equal_error_rate_from_ivecs


def factory(scorer):
    """
    gets a scorer class

    Args:
        scorer: the scorer type

    Returns:
        a scorer class
    """
    if scorer == 'eer' or scorer == 'EER':
        return equal_error_rate.EER
    elif scorer == 'eer_from_ivec':
        return equal_error_rate_from_ivecs.EERFromIvecs
    else:
        raise Exception('Undefined scorer type: %s' % scorer)
