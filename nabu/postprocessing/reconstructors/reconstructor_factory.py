'''@file reconstructor_factory.py
contains the Reconstructor factory'''

from . import  deepclustering_reconstructor, stackedmasks_reconstructor, \
deepXclustering_reconstructor, pit_l41_reconstructor, deepclustering_tf_reconstructor, \
parallel_deepclustering_reconstructor

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
    elif reconstructor == 'deepclusteringtf':
        return deepclustering_tf_reconstructor.DeepclusteringTFReconstructor
    elif reconstructor == 'deepXclustering':
        return deepXclustering_reconstructor.DeepXclusteringReconstructor  
    elif reconstructor == 'stackedmasks':
        return stackedmasks_reconstructor.StackedmasksReconstructor 
    elif reconstructor == 'pit_l41':
        return pit_l41_reconstructor.PITL41Reconstructor
    elif reconstructor == 'parallel_deepclustering':
        return parallel_deepclustering_reconstructor.ParallelDeepclusteringReconstructor
    else:
        raise Exception('Undefined reconstructor type: %s' % reconstructor)