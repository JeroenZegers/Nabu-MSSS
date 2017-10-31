'''@file reconstructor_factory.py
contains the Reconstructor factory'''

from . import  deepclustering_reconstructor, stackedmasks_reconstructor, \
deepXclustering_reconstructor

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
    if reconstructor == 'deepXclustering':
        return deepXclustering_reconstructor.DeepXclusteringReconstructor  
    elif reconstructor == 'stackedmasks':
        return stackedmasks_reconstructor.StackedmasksReconstructor
    else:
        raise Exception('Undefined reconstructor type: %s' % reconstructor)