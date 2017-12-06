'''@file reconstructor_factory.py
contains the Reconstructor factory'''

from . import  deepclustering_reconstructor, stackedmasks_reconstructor, \
deepXclustering_reconstructor, deepattractornet_reconstructor, pit_l41_reconstructor


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
    elif reconstructor == 'deepXclustering':
        return deepXclustering_reconstructor.DeepXclusteringReconstructor  
    elif reconstructor == 'stackedmasks':
        return stackedmasks_reconstructor.StackedmasksReconstructor 
    elif reconstructor == 'pit_l41':
        return pit_l41_reconstructor.PITL41Reconstructor
    else:
        raise Exception('Undefined reconstructor type: %s' % reconstructor)
