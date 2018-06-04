'''@file loss_computer_factory.py
contains the Loss computer factory mehod'''


from . import deepclustering_loss, pit_loss, l41_loss, pit_l41_loss,deepattractornet_sigmoid_loss, \
deepclusteringnoise_loss,deepattractornetnoise_hard_loss,deepattractornetnoise_soft_loss, \
deepattractornet_softmax_loss, noisefilter_loss, deepattractornet_noisefilter_loss, \
nb_speakers_loss


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
    elif loss_type == 'deepattractornet_sigmoid':
        return deepattractornet_sigmoid_loss.DeepattractornetSigmoidLoss
    elif loss_type == 'deepattractornet_softmax':
        return deepattractornet_softmax_loss.DeepattractornetSoftmaxLoss
    elif loss_type == 'l41':
        return l41_loss.L41Loss
    elif loss_type == 'pit_l41':
        return pit_l41_loss.PITL41Loss
    elif loss_type == 'deepclusteringnoise':
        return deepclusteringnoise_loss.DeepclusteringnoiseLoss
    elif loss_type == 'deepattractornetnoisehard':
        return deepattractornetnoise_hard_loss.DeepattractornetnoisehardLoss
    elif loss_type == 'deepattractornetnoisesoft':
        return deepattractornetnoise_soft_loss.DeepattractornetnoisesoftLoss
    elif loss_type == 'noisefilter':
        return noisefilter_loss.NoisefilterLoss
    elif loss_type == 'deepattractornet_noisefilter':
        return deepattractornet_noisefilter_loss.DeepattractornetnoisefilterLoss
    elif loss_type = 'nb_speakers':
        return nb_speakers_loss.NbSpeakerLoss
    else:
        raise Exception('Undefined loss type: %s' % loss_type)
