'''@file loss_computer_factory.py
contains the Loss computer factory mehod'''

from . import deepclustering_loss, pit_loss, l41_loss, pit_l41_loss

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
    elif loss_type == 'l41':
        return l41_loss.L41Loss
    elif loss_type == 'pit_l41':
        return pit_l41_loss.PITL41Loss
    else:
        raise Exception('Undefined loss type: %s' % loss_type)
