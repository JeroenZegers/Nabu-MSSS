'''@file scorer_factory.py
contains the Scorer factory'''

from . import  sdr_scorer

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
    else:
        raise Exception('Undefined scorer type: %s' % scorer)
