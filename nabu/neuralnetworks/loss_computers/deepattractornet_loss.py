'''@file deepclustering_loss.py
contains the DeepclusteringLoss'''

import tensorflow as tf
import loss_computer
from nabu.neuralnetworks.components import ops

class DeepattractornetLoss(loss_computer.LossComputer):
    '''A loss computer that calculates the loss'''

    def __call__(self, targets, embeddings,mixture, seq_length):

        '''
        Compute the loss 

        Creates the operation to compute the deep clustering loss

        Args:
            targets: a dictionary of [batch_size x time x ... ] tensor containing
                the targets
            embeddings: a dictionary of [batch_size x time x (feature_dim*embedding_dim)] tensor containing
                the embeddings
            mixture: a [batch_size x (time * feature_dim)] tensor containing the spectograms of the mixtures
            seq_length: a dictionary of [batch_size] vectors containing
                the sequence lengths

        Returns:
            a scalar value containing the loss
        '''
                   
        outputs = embeddings['outputs']
        binary_target=targets['partition_targets']

        multi_targets=targets['spectogram_targets']
        # Spectogram of the mixture, used to mask
        mix_to_mask = targets['mix_to_mask']
        usedbins = targets['usedbins']
        seq_length = seq_length['features']
		    
        loss = ops.deepattractornet_loss(binary_target, multi_targets, mix_to_mask, usedbins, outputs,seq_length,self.batch_size)
        return loss
