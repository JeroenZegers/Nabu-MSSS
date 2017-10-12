'''@file loss_computer_factory.py
contains the Loss computer factory mehod'''

from . import deepclustering_loss, pit_loss,deepattractornet_loss

def factory(loss_type):
    '''gets a Loss computer class

    Args:
        loss_type: the loss type

    Returns: a Loss computer class
    '''

    if loss_type == 'deepclustering':
        return deepclustering_loss.DeepclusteringLoss
    elif loss_type == 'pit':
        return pit_loss.PITLoss
    elif loss_type == 'deepattractornet':
        return deepattactornet_loss.DeepattractornetLoss
    else:
        raise Exception('Undefined loss type: %s' % loss_type)
