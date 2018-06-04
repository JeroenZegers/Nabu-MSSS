'''@file ops.py
some operations'''

import tensorflow as tf
import itertools
import pdb

def pyramid_stack(inputs, sequence_lengths, numsteps, axis=2, scope=None):
    '''
    concatenate each two consecutive elements

    Args:
        inputs: A time minor tensor [batch_size, time, input_size]
        sequence_lengths: the length of the input sequences
        numsteps: number of time steps to concatenate
        axis: the axis where the inputs should be stacked
        scope: the current scope

    Returns:
        inputs: Concatenated inputs
            [batch_size, time/numsteps, input_size*numsteps]
        sequence_lengths: the lengths of the inputs sequences [batch_size]
    '''

    with tf.name_scope(scope or 'pyramid_stack'):

        numdims = len(inputs.shape)

        #convert imputs to time major
        time_major_input = tf.transpose(inputs, [1, 0] + range(2, numdims))


        #pad the inputs to an appropriate length length
        length = tf.cast(tf.shape(time_major_input)[0], tf.float32)
        pad_length = tf.ceil(length/numsteps)*numsteps - length
        pad_length = tf.cast(pad_length, tf.int32)
        pad_shape = tf.concat([[pad_length],
                               tf.shape(time_major_input)[1:]], 0)
        padding = tf.zeros(pad_shape, dtype=inputs.dtype)
        padded_inputs = tf.concat([time_major_input, padding], 0)

        #get the new length
        length = tf.shape(padded_inputs)[0]

        #seperate the inputs for every concatenated timestep
        seperated = []
        for i in range(numsteps):
            seperated.append(tf.gather(
                padded_inputs, tf.range(i, length, numsteps)))

        #concatenate odd and even inputs
        time_major_outputs = tf.concat(seperated, axis)

        #convert back to time minor
        outputs = tf.transpose(time_major_outputs, [1, 0] + range(2, numdims))

        #compute the new sequence length
        output_sequence_lengths = tf.cast(tf.ceil(tf.cast(sequence_lengths,
                                                          tf.float32)/numsteps),
                                          tf.int32)

    return outputs

def seq2nonseq(sequential, sequence_lengths, name=None):
    '''
    Convert sequential data to non sequential data

    Args:
        sequential: the sequential data which is a [batch_size, max_length, dim]
            tensor
        sequence_lengths: a [batch_size] vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths
    '''

    with tf.name_scope(name or 'seq2nonseq'):

        indices = get_indices(sequence_lengths)

        #create the values
        tensor = tf.gather_nd(sequential, indices)


    return tensor

def dense_sequence_to_sparse(sequences, sequence_lengths):
    '''convert sequence dense representations to sparse representations

    Args:
        sequences: the dense sequences as a [batch_size x max_length] tensor
        sequence_lengths: the sequence lengths as a [batch_size] vector

    Returns:
        the sparse tensor representation of the sequences
    '''

    with tf.name_scope('dense_sequence_to_sparse'):

        #get all the non padding sequences
        indices = tf.cast(get_indices(sequence_lengths), tf.int64)

        #create the values
        values = tf.gather_nd(sequences, indices)

        #the shape
        shape = tf.cast(tf.shape(sequences), tf.int64)

        sparse = tf.SparseTensor(indices, values, shape)

    return sparse

def deepattractornet_sigmoid_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings, \
    seq_length, batch_size):
    '''
    Compute the deep attractor net loss (as described in Deep attractor network for single-microphone speaker separation,
        Zhuo Chen, et al. [1]) using sigmoid masks.

    Args:
        partitioning:       optimal binary partitioning of the bins         [batch_size x time x (feature_dim(F)*nrS)]
        spectogram_targets: clean spectograms of the sources                [batch_size x time x feature_dim (F)  x nrS]
        mix_to_mask:        spectograms of the mixture                      [batch_size x time x feature_dim (F)]
        energybins:         indicating whether the bin has enough energy    [batch_size x time x feature_dim (F)]
        embeddings:         resulting embeddingsvectors                     [batch_size x time x (feature_dim(F) * emb_dim (D)] tensor containing the
        seq_length:         sequence lengths                                [batch_size]
        batch_size:         number of utterances in this batch
    Returns:
        loss: a scalar value containing the loss
        norm: a scalar value containing the normalisation constant
    '''
    with tf.name_scope('deepattractornet_sigmoid_loss'):
        # Number of frequency bins
        F = tf.shape(usedbins)[2]
        # Dimension embeddingspace
        D = tf.shape(embeddings)[2]/F
        # Number of sources
        nrS= tf.shape(spectogram_targets)[3]

        # Initialise loss and normalisation with zero
        loss = 0.0
        norm = 0.0

        # Loop over utterances in this batch
        for utt_ind in range(batch_size):
            # Number of timeframes in this utterance
            T = seq_length[utt_ind]
            # Number of time-frequency bins in current spectogram
            K = T*F
            # Which time-frequency bins have enough enery
            energybins_utt = energybins[utt_ind]
            energybins_utt = energybins_utt[:T,:] # dim: (Tx F)
            # Embedding vectors of this utterance
            embedding_utt = embeddings[utt_ind]
            embedding_utt = embedding_utt[:T,:] # dim: (T x F*D)
            # Partitioning of this utterance
            partition_utt = partitioning[utt_ind]
            partition_utt = partition_utt[:T,:] # dim: (T x F*nrS)
            # Spectrogram of this utterance
            mix_to_mask_utt = mix_to_mask[utt_ind]
            mix_to_mask_utt = mix_to_mask_utt[:T,:] # dim: (T x F)
            # Clean spectogram of the speakers in this utterance
            spectogram_utt = spectogram_targets[utt_ind]
            spectogram_utt = spectogram_utt[:T,:,:] # dim: (T x F x nrS)

            # Reshape the energy indication
            ebresh = tf.reshape(energybins_utt,[K,1],name='ebresh') # dim: (K x 1)
            ebreshY=tf.tile(ebresh,[1,nrS]) # dim: (K x nrS)

            # Reshape the embeddingsvectors
            V = tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

            # Reshape partitioning and remove silent bins
            Y = tf.reshape(partition_utt,[K,nrS],name='Y')
            Y = tf.multiply(Y,ebreshY)
            Y = tf.to_float(Y) # dim: (K x nrS)

            # Calculate attractor centers
            numerator_A=tf.matmul(Y,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
            # Number of bins each speaker dominates
            nb_bins_class = tf.reduce_sum(Y,axis = 0) # dim: (rank 1) (nrS)
            # Set number of bins of each speaker to at least 1 to avoid division by zero
            nb_bins_class = tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
            nb_bins_class = tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
            denominator_A = tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
            A = tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

            # Calculate sigmoid mask
            prod_1 = tf.matmul(A,V,transpose_a=False, transpose_b = True,name='AVT')
            M = tf.sigmoid(prod_1,name='M') # dim: (nrS x K)

            # Reshape spectrogram of mixture
            X = tf.transpose(tf.reshape(mix_to_mask_utt,[K,1],name='X')) # dim: (1 x K)
            # Calculate the reconstructions
            reconstructions = tf.multiply(M,X) # dim: (nrS x K)
            S = tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
            loss_utt = tf.reduce_sum(tf.square(S-reconstructions),name='loss')
            # update loss and normalisation
            norm += tf.to_float(K*nrS)
            loss += loss_utt

        return loss,norm

def deepattractornet_softmax_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings,\
                seq_length, batch_size):
    '''
    Compute the deep attractor net loss (as described in Deep attractor network for single-microphone speaker separation,
        Zhuo Chen, et al. [1]) with softmax masks.

    Args:
        partitioning:       optimal binary partitioning of the bins         [batch_size x time x (feature_dim(F)*nrS)]
        spectogram_targets: clean spectograms of the sources                [batch_size x time x feature_dim (F)  x nrS]
        mix_to_mask:        spectograms of the mixture                      [batch_size x time (T) x feature_dim (F)]
        energybins:         indicating whether the bin has enough energy    [batch_size x time(T) x feature_dim (F)]
        embeddings:         resulting embeddingsvectors                     [batch_size x time (T) x (feature_dim(F) * emb_dim (D)] tensor containing the
        seq_length:         sequence lengths                                [batch_size]
        batch_size:         number of utterances in this batch
    Returns:
        loss: a scalar value containing the loss
        norm: a scalar value containing the normalisation constant
    '''
    with tf.name_scope('deepattractornet_softmax_loss'):
        # Number of frequency bins
        F = tf.shape(usedbins)[2]
        # Dimension embeddingspace
        D = tf.shape(embeddings)[2]/F
        # Number of sources
        nrS= tf.shape(spectogram_targets)[3]

        # Initialise loss and normalisation with zero
        loss = 0.0
        norm = 0.0

        # Loop over utterances in this batch
        for utt_ind in range(batch_size):
            # Number of timeframes in this utterance
            T = seq_length[utt_ind]
            # Number of time-frequency bins in current spectogram
            K = T*F
            # Which time-frequency bins have enough enery
            energybins_utt = energybins[utt_ind]
            energybins_utt = energybins_utt[:T,:] # dim: (Tx F)
            # Embedding vectors of this utterance
            embedding_utt = embeddings[utt_ind]
            embedding_utt = embedding_utt[:T,:] # dim: (T x F*D)
            # Partitioning of this utterance
            partition_utt = partition_targets[utt_ind]
            partition_utt = partition_utt[:T,:] # dim: (T x F*nrS)
            # Spectrogram of this utterance
            mix_to_mask_utt = mix_to_mask[utt_ind]
            mix_to_mask_utt = mix_to_mask_utt[:T,:] # dim: (T x F)
            # Clean spectogram of the speakers in this utterance
            spectogram_utt = spectogram_targets[utt_ind]
            spectogram_utt = spectogram_utt[:T,:,:] # dim: (T x F x nrS)

            # Reshape the energy indication
            ebresh = tf.reshape(energybins_utt,[K,1],name='ebresh') # dim: (K x 1)
            ebreshY=tf.tile(ebresh,[1,nrS]) # dim: (K x nrS)

            # Reshape the embeddingsvectors
            V = tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

            # Reshape partitioning and remove silent bins
            Y = tf.reshape(partition_utt,[K,nrS],name='Y') # dim: N x number_sources
            Y = tf.multiply(Y,ebreshY)
            Y = tf.to_float(Y) # dim: (K x nrS)

            # Calculate attractor centers
            numerator_A = tf.matmul(Y,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
            # Number of bins each speaker dominates
            nb_bins_class = tf.reduce_sum(Y,axis = 0) # dim: (rank 1) (nrS)
            # Set number of bins of each speaker to at least 1 to avoid division by zero
            nb_bins_class = tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
            nb_bins_class = tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
            denominator_A = tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
            A = tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

            # Calculate softmax masker
            prod_1 = tf.matmul(A,V,transpose_a=False, transpose_b = True,name='AVT')
            M = tf.nn.softmax(prod_1,dim = 0,name='M') # dim: (nrS x K)

            # Reshape spectrogram of mixture
            X = tf.transpose(tf.reshape(mix_to_mask_utt,[K,1],name='X')) # dim: (1 x K)
            # Calculate the reconstructions
            reconstructions = tf.multiply(M,X) # dim: (nrS x K)
            S = tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
            # Calculate loss of this utterance
            loss_utt = tf.reduce_sum(tf.square(S-reconstructions),name='loss') # Calculate difference reconstructions and clean spectrogram
            # update loss and normalisation
            norm += tf.to_float(N*nr_S)
            loss += loss_utt

        return loss,norm

def deepattractornetnoise_soft_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings, \
        alpha, seq_length, batch_size):
    '''
    Compute the deep attractor net loss (as described in Deep attractor network for single-microphone speaker separation,
        Zhuo Chen, et al. [1]) with adapted architecture for noise and soft decissions

    Args:
        partitioning:       optimal binary partitioning of the bins         [batch_size x time x (feature_dim(F)*nrS)]
        spectogram_targets: clean spectograms of the sources                [batch_size x time x feature_dim (F)  x nrS]
        mix_to_mask:        spectograms of the mixture                      [batch_size x time x feature_dim (F)]
        energybins:         indicating whether the bin has enough energy    [batch_size x time x feature_dim (F)]
        embeddings:         resulting embeddingsvectors                     [batch_size x time x (feature_dim(F) * emb_dim (D)]
        alpha:              resulting alpha                                 [batch_size x time x feature_dim (F)]
        seq_length:         sequence lengths                                [batch_size]
        batch_size:         number of utterances in this batch
    Returns:
        loss: a scalar value containing the loss
        norm: a scalar value containing the normalisation constant

    '''
    with tf.name_scope('deepattractornetnoise_soft_loss'):
        # Number of frequency bins
        F = tf.shape(usedbins)[2]
        # Dimension embeddingspace
        D = tf.shape(embeddings)[2]/F
        # Number of sources
        nrS= tf.shape(spectogram_targets)[3]

        # Initialise loss and normalisation with zero
        loss = 0.0
        norm = 0.0

        # Loop over utterances in this batch
        for utt_ind in range(batch_size):
            # Number of timeframes in this utterance
            T = seq_length[utt_ind]
            # Number of time-frequency bins in current spectogram
            K = T*F
            # Which time-frequency bins have enough enery
            energybins_utt = energybins[utt_ind]
            energybins_utt = energybins_utt[:T,:]  # dim: (T x F)
            # Embedding vectors of this utterance
            embedding_utt = embeddings[utt_ind]
            embedding_utt = embedding_utt[:T,:] # dim: (T x F*D)
            # Partitioning of this utterance
            partition_utt = partitioning[utt_ind]
            partition_utt = partition_utt[:T,:] # dim: (T x F*nrS)
            # Spectrogram of this utterance
            mix_to_mask_utt = mix_to_mask[utt_ind]
            mix_to_mask_utt = mix_to_mask_utt[:T,:] # dim: (T x F)
            # Clean spectogram of the speakers in this utterance
            spectogram_utt = spectogram_targets[utt_ind]
            spectogram_utt = spectogram_utt[:T,:,:] # dim: (T x F x nrS)
            # Alpha outputs of the network
            alpha_utt = alpha[utt_ind]
            alpha_utt = alpha_utt[:T,:] # dim: (TxF)

            # Reshape the energy indication
            ebresh = tf.reshape(energybins_utt,[K,1],name='ubresh') # dim: (K x 1)
            ebreshY=tf.tile(ebresh,[1,nrS],name='ubreshY') # dim: (K x nrS)

            # Value of alpha as weight for bin in calculation attractor
            alpharesh = tf.reshape(alpha_utt,[K,1],name='nbresh') # dim: (K x 1)
            alphareshY = tf.tile(alpharesh,[1,nrS],name='nbreshY') # dim: (K x nrS)

            # Reshape the embeddingsvectors
            V = tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

            # Reshape partitioning, remove silent bins and apply weights
            Y = tf.reshape(partition_utt,[K,nrS],name='Y')
            Y_tilde = tf.to_float(tf.multiply(Y,ebreshY),name='Y_tilde') # dim: (K x nrS)
            Y_tilde_W = tf.multiply(Y_tilde,alphareshY,name='Y_tilde_W') # dim: (K x nrS)

            # Calculate attractor centers
            numerator_A=tf.matmul(Y_tilde_W,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
            nb_bins_class = tf.reduce_sum(Y_tilde_W,axis = 0) # dim: (rank 1) (nrS)
            # Set number of bins of each speaker to at least 1 to avoid division by zero
            nb_bins_class = tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
            nb_bins_class = tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
            denominator_A = tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
            A = tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

            # Calculate softmax mask
            prod_1 = tf.matmul(A,V,transpose_a=False, transpose_b = True,name='AVT')
            M_speaker = tf.nn.softmax(prod_1,dim = 0,name='M') # dim: (nrS x K)

            # Reshape spectrogram of mixture
            X = tf.reshape(mix_to_mask_utt,[K,1],name='X') # dim: (K x 1)
            # Filter noise from mixture
            X_filter_noise = tf.transpose(tf.multiply(X,alpharesh)) # dim: (1 x K)
            reconstructions = tf.multiply(M_speaker,X_filter_noise) # dim: (nrS x K)
            S = tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
            loss_utt = tf.reduce_sum(tf.square(S-reconstructions),name='loss')

            # update loss and normalisation
            norm += tf.to_float(K*nrS)
            loss += loss_utt

        return loss,norm

def deepattractornetnoise_hard_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings, \
            alpha, seq_length, batch_size):
    '''
    Compute the deep attractor net loss (as described in Deep attractor network for single-microphone speaker separation,
        Zhuo Chen, et al. [1]) with adapted architecture for noise and hard decissions

    Args:
        partitioning:       optimal binary partitioning of the bins         [batch_size x time x (feature_dim(F)*nrS)]
        spectogram_targets: clean spectograms of the sources                [batch_size x time x feature_dim (F)  x nrS]
        mix_to_mask:        spectograms of the mixture                      [batch_size x time x feature_dim (F)]
        energybins:         indicating whether the bin has enough energy    [batch_size x time x feature_dim (F)]
        embeddings:         resulting embeddingsvectors                     [batch_size x time x (feature_dim(F) * emb_dim (D)] tensor containing the
        alpha:              resulting alpha                                 [batch_size x time x feature_dim (F)]
        seq_length:         sequence lengths                                [batch_size]
        batch_size:         number of utterances in this batch
    Returns:
        loss: a scalar value containing the loss
        norm: a scalar value containing the normalisation constant
    '''
    with tf.name_scope('deepattractornetnoise_hard_loss'):
        # Number of frequency bins
        F = tf.shape(usedbins)[2]
        # Dimension embeddingspace
        D = tf.shape(embeddings)[2]/F
        # Number of sources
        nrS= tf.shape(spectogram_targets)[3]

        # Initialise loss and normalisation with zero
        loss = 0.0
        norm = 0.0
        # Only time frequency bins with alpha above this threshold are used for calculating the attractorpoints
        noise_threshold = 0.75
        # Loop over utterances in this batch
        for utt_ind in range(batch_size):
            # Number of timeframes in this utterance
            T = seq_length[utt_ind]
            # Number of time-frequency bins in current spectogram
            K = T*F
            # Which time-frequency bins have enough enery
            energybins_utt = usedbins[utt_ind]
            energybins_utt = usedbins_utt[:T,:] # dim: (Tx F)
            # Embedding vectors of this utterance
            embedding_utt = embeddings[utt_ind]
            embedding_utt = embedding_utt[:T,:] # dim: (T x F*D)
            # Partitioning of this utterance
            partition_utt = partitioning[utt_ind]
            partition_utt = partition_utt[:T,:] # dim: (T x F*nrS)
            # Spectrogram of this utterance
            mix_to_mask_utt = mix_to_mask[utt_ind]
            mix_to_mask_utt = mix_to_mask_utt[:T,:] # dim: (T x F)
            # Clean spectogram of the speakers in this utterance
            spectogram_utt = spectogram_targets[utt_ind]
            spectogram_utt = spectogram_utt[:T,:,:] # dim: (T x F x nrS)
            # Alpha outputs of the network
            alpha_utt = alpha[utt_ind]
            alpha_utt = alpha_utt[:T,:] # dim: (TxF)

            # Reshape the energy indication
            ebresh = tf.reshape(energybins_utt,[K,1],name='ubresh')# dim: (K x 1)
            ebreshY=tf.tile(ebresh,[1,nrS]) # dim: (K x nrS)
            # Threshold and reshape alpha
            alpharesh = tf.cast(tf.reshape(alpha_utt,[K,1])>noise_threshold,tf.int32)
            alphareshY = tf.tile(nbresh,[1,nrS])

            # Reshape the embeddingsvectors
            V = tf.reshape(embedding_utt,[N,emb_dim],name='V') # dim: (K x D)

            # Reshape partitioning, remove silent and noisy bins
            Y = tf.reshape(partition_utt,[N,nr_S],name='Y') # dim: (K x 1)
            Y_tilde_tilde = tf.multiply(tf.multiply(Y,ubreshY),nbreshY) # dim: (K x nrS)
            Y_tilde_tilde = tf.to_float(Y_tilde_tilde) # dim: (K x nrS)

            # Calculate attractor centers
            numerator_A=tf.matmul(Y_tilde_tilde,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
            nb_bins_class = tf.reduce_sum(Y_tilde_tilde,axis = 0) # dim: (rank 1) (nrS)
            # Set number of bins of each speaker to at least 1 to avoid division by zero
            nb_bins_class = tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
            nb_bins_class = tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
            denominator_A = tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
            A = tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

            # Calculate softmax mask
            prod_1 = tf.matmul(A,V,transpose_a=False, transpose_b = True,name='AVT')

            M_speaker = tf.nn.softmax(prod_1,dim = 0,name='M') # dim: (nrS x K)
            M_noise = tf.transpose(tf.reshape(alpha_utt,[K,1])) # dim: (K x 1)

            # Reshape spectrogram of mixture
            X = tf.transpose(tf.reshape(mix_to_mask_utt,[N,1],name='X')) # dim: (1 x K)
            # Filter noise from mixture
            X_filter_noise = tf.multiply(X,M_noise) # dim: (1 x K)
            reconstructions = tf.multiply(M_speaker,X_filter_noise) # dim: number_sources x N
            S = tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K])  # dim: (nrS x K)
            loss_utt = tf.reduce_sum(tf.square(S-reconstructions),name='loss')  # dim: (nrS x K)
            # update loss and normalisation
            norm += tf.to_float(K*nrS)
            loss += loss_utt

        return loss,norm

def noise_filter_loss(clean_spectrogram,noise_spectrogram,alpha,seq_length,batch_size):
    '''
        Calculates loss for loss filter
        Args:
            clean_spectrogram:  clean (target) spectrogram       [batch_size x time x feature_dim (F)]
            noise_spectogram:   noisy spectrogram                [batch_size x time x feature_dim (F)]
            alpha:              resulting alpha                  [batch_size x time x feature_dim (F)]
            seq_length:         sequence lengths                 [batch_size]
            batch_size:         number of utterances in this batch
        Returns:
            a scalar value containing the loss
            a scalar value containing the normalisation constant
    '''
    with tf.name_scope('noise_filter_loss'):
        # Number of frequency bins
        F = tf.shape(noise_spectrogram)[2]
        # Initialise loss and normalisation with zero
        loss = 0.0
        norm = 0.0
        # Loop over utterances in this batch
        for utt_ind in range(batch_size):
            # Number of timeframes in this utterance
            T = seq_length[utt_ind]
            # Noisy spectrogram
            noise_spectrogram_utt = noise_spectrogram[utt_ind]
            noise_spectrogram_utt = noise_spectrogram_utt[:T,:] # dim: (T x F)
            # Target spectrogram
            clean_spectrogram_utt = clean_spectrogram[utt_ind]
            clean_spectrogram_utt = clean_spectrogram_utt[:T,:] # dim: (T x F)

            # Alpha outputs of the network
            alpha_utt = alpha[utt_ind]
            alpha_utt = alpha_utt[:T,:] # dim: (T x F)

            estimate = tf.multiply(noise_spectrogram_utt,alpha_utt) # dim: (T x F)
            # update loss and normalisation
            loss += tf.reduce_sum(tf.square(clean_spectrogram_utt-estimate),name='loss')
            norm += tf.to_float(T*F)
    return loss,norm

def deepattractornet_noisefilter_loss(partitioning, spectrogram_targets, mix_to_mask,\
    embeddings,alpha,seq_length,batch_size):
    '''
    Compute the deep attractor net loss (as described in Deep attractor network for single-microphone speaker separation,
        Zhuo Chen, et al. [1]) with adapted architecture for noise and soft decissions

    Args:
        partitioning:       optimal binary partitioning of the bins         [batch_size x time x (feature_dim(F)*nrS)]
        spectogram_targets: clean spectograms of the sources                [batch_size x time x feature_dim (F)  x nrS]
        mix_to_mask:        spectograms of the mixture                      [batch_size x time x feature_dim (F)]
        embeddings:         resulting embeddingsvectors                     [batch_size x time x (feature_dim(F) * emb_dim (D)] tensor containing the
        alpha:              resulting alpha                                 [batch_size x time x feature_dim (F)]
        seq_length:         sequence lengths                                [batch_size]
        batch_size:         number of utterances in this batch
    Returns:
        loss: a scalar value containing the loss
        norm: a scalar value containing the normalisation constant

    '''
    with tf.name_scope('deepattractornet_noisefilter'):
        # Number of frequency bins
        F = tf.shape(mix_to_mask)[2]
        # Dimension embeddingspace
        D = tf.shape(embeddings)[2]/F
        # Number of sources
        nrS= tf.shape(spectrogram_targets)[3]

        # Initialise loss and normalisation with zero
        loss = 0.0
        norm = 0.0
        # Threshold on energy
        usedbin_threshold = 100.
        # Loop over utterances in this batch
        for utt_ind in range(batch_size):
            # Number of timeframes in this utterance
            T = seq_length[utt_ind]
            # Number of time-frequency bins in current spectogram
            K = T*F
            # Embedding vectors of this utterance
            embedding_utt = embeddings[utt_ind]
            embedding_utt = embedding_utt[:T,:] # dim: (T x F*D)
            # Partitioning of this utterance
            partition_utt = partitioning[utt_ind]
            partition_utt = partition_utt[:T,:] # dim: (T x F*nrS)
            # Spectrogram of this utterance
            mix_to_mask_utt = mix_to_mask[utt_ind]
            mix_to_mask_utt = mix_to_mask_utt[:T,:] # dim: (T x F)
            # Clean spectogram of the speakers in this utterance
            spectrogram_utt = spectrogram_targets[utt_ind]
            spectrogram_utt = spectrogram_utt[:T,:,:] # dim: (T x F x nrS)

            alpha_utt = alpha[utt_ind]
            alpha_utt = alpha_utt[:T,:] # dim: (TxF)

            # Eliminate noise noise
            X_hat_clean = tf.reshape(tf.multiply(mix_to_mask_utt,alpha_utt),[K,1]) # dim: (T x F)

            # Calculate bins with enough energy in cleaned spectrogram
            maxbin = tf.reduce_max(X_hat_clean)
    	    floor = maxbin/usedbin_threshold
    	    ubresh = tf.cast(tf.greater(X_hat_clean,floor),tf.int32)
            ubreshY=tf.tile(ubresh,[1,nr_S]) # dim: (K x nrS)

            # Reshape the embeddingsvectors
            V = tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

            # Reshape partitioning and remove silent bins
            Y = tf.reshape(partition_utt,[K,nrS],name='Y')
            Y_tilde = tf.multiply(Y,ubreshY)
            Y_tilde = tf.to_float(Y_tilde)

            # Calculate attractor centers
            numerator_A=tf.matmul(Y_tilde,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
            nb_bins_class = tf.reduce_sum(Y_tilde,axis = 0) # dim: (rank 1) (nrS)
            # Set number of bins of each speaker to at least 1 to avoid division by zero
            nb_bins_class = tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
            nb_bins_class = tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
            denominator_A = tf.tile(nb_bins_class,[1,emb_dim],name='denominator_A') # dim: (nrS x D)
            A = tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

            # Calculate softmax mask
            prod_1 = tf.matmul(A,V,transpose_a=False, transpose_b = True,name='AVT')
            M_speaker = tf.nn.softmax(prod_1,dim = 0,name='M') # dim: number_sources x N
            # Calculate reconstructions
            reconstructions = tf.multiply(M_speaker,tf.transpose(X_hat_clean)) # dim: (nrS x K)
            S = tf.reshape(tf.transpose(spectrogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
            loss_utt = tf.reduce_sum(tf.square(S-reconstructions),name='loss')
            # update loss and normalisation
            norm += tf.to_float(K*nrS)
            loss += loss_utt

        return loss,norm

def L41_loss(targets, bin_embeddings, spk_embeddings, usedbins, seq_length, batch_size):
    '''
    Monaural Audio Speaker Separation Using Source-Contrastive Estimation
    Cory Stephenson, Patrick Callier, Abhinav Ganesh, and Karl Ni

    Args:
        targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
        bin_embeddings: a [batch_size x time x (feat_dim*emb_dim)] tensor containing
        the timefrequency bin embeddings
        spk_embeddings: a [batch_size x 1 x (emb_dim*nrS))] tensor containing the speaker embeddings
        usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
        seq_length: a [batch_size] vector containing the
            sequence lengths
        batch_size: the batch size

    Returns:
        a scalar value containing the loss
    '''

    with tf.name_scope('L41_loss'):
	feat_dim = tf.shape(usedbins)[2]
        output_dim = tf.shape(bin_embeddings)[2]
        emb_dim = output_dim/feat_dim
        target_dim = tf.shape(targets)[2]
        nrS = target_dim/feat_dim

        loss = 0.0
        norm = 0

        for utt_ind in range(batch_size):
	    N = seq_length[utt_ind]
	    usedbins_utt = usedbins[utt_ind]
	    usedbins_utt = usedbins_utt[:N,:]
	    bin_emb_utt = bin_embeddings[utt_ind]
	    bin_emb_utt = bin_emb_utt[:N,:]
	    targets_utt = targets[utt_ind]
	    targets_utt = targets_utt[:N,:]
	    spk_emb_utt = spk_embeddings[utt_ind]

	    vi = tf.reshape(bin_emb_utt,[N,feat_dim,1,emb_dim],name='vi')
	    vi_norm = tf.nn.l2_normalize(vi,3,name='vi_norm')
	    vo = tf.reshape(spk_emb_utt,[1,1,nrS,emb_dim],name='vo')
	    vo_norm = tf.nn.l2_normalize(vo,3,name='vo_norm')

	    dot = tf.reduce_sum(vi_norm*vo_norm,3,name='D')

	    Y = tf.to_float(tf.reshape(targets_utt,[N,feat_dim,nrS]))
	    Y = (Y-0.5)*2.0

	    # Compute the cost for every element
	    loss_utt = -tf.log(tf.nn.sigmoid(Y * dot))

	    loss_utt = tf.reduce_sum(tf.to_float(tf.expand_dims(usedbins_utt,-1))*loss_utt)

	    loss += loss_utt

	    norm += tf.to_float(tf.reduce_sum(usedbins_utt)*nrS)

    #loss = loss/tf.to_float(batch_size)

    return loss , norm

def pit_L41_loss(targets, bin_embeddings, spk_embeddings, mix_to_mask, seq_length, batch_size):
    '''
    Combination of L41 approach, where an attractor embedding per speaker is found and PIT
    where the audio signals are reconstructed via mast estimation, which are used to define
    a loss in a permutation invariant way. Here the masks are estimated by evaluating the distance
    of a bin embedding to all speaker embeddings.

    Args:
        targets: a [batch_size x time x feat_dim  x nrS)] tensor containing the multiple targets
        bin_embeddings: a [batch_size x time x (feat_dim*emb_dim)] tensor containing
        the timefrequency bin embeddings
        spk_embeddings: a [batch_size x 1 x (emb_dim*nrS)] tensor containing the speaker embeddings
        mix_to_mask: a [batch_size x time x feat_dim] tensor containing the mixture that will be masked
        seq_length: a [batch_size] vector containing the
            sequence lengths
        batch_size: the batch size

    Returns:
        a scalar value containing the loss
    '''

    with tf.name_scope('PIT_L41_loss'):
	feat_dim = tf.shape(targets)[2]
        output_dim = tf.shape(bin_embeddings)[2]
        emb_dim = output_dim/feat_dim
        target_dim = tf.shape(targets)[2]
        nrS = targets.get_shape()[3]
        nrS_tf = tf.shape(targets)[3]
        permutations = list(itertools.permutations(range(nrS),nrS))

        loss = 0.0
        norm = tf.to_float(nrS_tf * feat_dim * tf.reduce_sum(seq_length))

        for utt_ind in range(batch_size):
	    N = seq_length[utt_ind]
	    bin_emb_utt = bin_embeddings[utt_ind]
	    bin_emb_utt = bin_emb_utt[:N,:]
	    targets_utt = targets[utt_ind]
	    targets_utt = targets_utt[:N,:,:]
	    spk_emb_utt = spk_embeddings[utt_ind]
	    mix_to_mask_utt = mix_to_mask[utt_ind]
	    mix_to_mask_utt = mix_to_mask_utt[:N,:]

	    vi = tf.reshape(bin_emb_utt,[N,feat_dim,1,emb_dim],name='vi')
	    vi_norm = tf.nn.l2_normalize(vi,3,name='vi_norm')
	    vo = tf.reshape(spk_emb_utt,[1,1,nrS_tf,emb_dim],name='vo')
	    vo_norm = tf.nn.l2_normalize(vo,3,name='vo_norm')

	    D = tf.divide(1,tf.norm(tf.subtract(vi_norm,vo_norm),ord=2,axis=3))
            Masks = tf.nn.softmax(D, dim=2)

	    #The masks are estimated, the remainder is the same as in pit_loss
	    mix_to_mask_utt = tf.expand_dims(mix_to_mask_utt,-1)
	    recs = tf.multiply(Masks, mix_to_mask_utt)

	    targets_resh = tf.transpose(targets_utt,perm=[2,0,1])
	    recs = tf.transpose(recs,perm=[2,0,1])

	    perm_cost = []
	    for perm in permutations:
		tmp = tf.square(tf.norm(tf.gather(recs,perm)-targets_resh,ord='fro',axis=[1,2]))
		perm_cost.append(tf.reduce_sum(tmp))

	    loss_utt = tf.reduce_min(perm_cost)

	    loss += loss_utt


    #loss = loss/tf.to_float(batch_size)

    return loss , norm

def deepclustering_loss(targets, logits, usedbins, seq_length, batch_size):
    '''
    Compute the deep clustering loss
    cost function based on Hershey et al. 2016

    Args:
        targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
        logits: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
        usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
        seq_length: a [batch_size] vector containing the sequence lengths
        batch_size: the batch size

    Returns:
        a scalar value containing the loss
    '''

    with tf.name_scope('deepclustering_loss'):
        feat_dim = tf.shape(usedbins)[2]
        output_dim = tf.shape(logits)[2]
        emb_dim = output_dim/feat_dim
        target_dim = tf.shape(targets)[2]
        nrS = target_dim/feat_dim

        loss = 0.0
        norm = 0.0

        for utt_ind in range(batch_size):
            N = seq_length[utt_ind]
            Nspec = N*feat_dim
            usedbins_utt = usedbins[utt_ind]
            usedbins_utt = usedbins_utt[:N,:]
            logits_utt = logits[utt_ind]
            logits_utt = logits_utt[:N,:]
            targets_utt = targets[utt_ind]
            targets_utt = targets_utt[:N,:]


            #remove the non_silence (cfr bins below energy thresh) bins. Removing in logits and
            #targets will give 0 contribution to loss.
            ubresh=tf.reshape(usedbins_utt,[Nspec,1],name='ubresh')
            ubreshV=tf.tile(ubresh,[1,emb_dim])
            ubreshV=tf.to_float(ubreshV)
            ubreshY=tf.tile(ubresh,[1,nrS])

            V=tf.reshape(logits_utt,[Nspec,emb_dim],name='V')
            Vnorm=tf.nn.l2_normalize(V, dim=1, epsilon=1e-12, name='Vnorm')
            Vnorm=tf.multiply(Vnorm,ubreshV)
            Y=tf.reshape(targets_utt,[Nspec,nrS],name='Y')
            Y=tf.multiply(Y,ubreshY)
            Y=tf.to_float(Y)

            prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, a_is_sparse=True,
	                b_is_sparse=True, name='VTV')
            prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, a_is_sparse=True,
	                b_is_sparse=True, name='VTY')

            term1=tf.reduce_sum(tf.square(prod1),name='frob_1')
            term2=tf.reduce_sum(tf.square(prod2),name='frob_2')

            loss_utt = tf.add(term1,-2*term2,name='term1and2')
            #normalizer= tf.to_float(tf.square(tf.reduce_sum(ubresh)))
            #loss += loss_utt/normalizer*(10**9)
            loss += loss_utt
            norm += tf.square(tf.to_float(tf.reduce_sum(usedbins_utt)))

    #loss = loss/tf.to_float(batch_size)

    return loss , norm

def deepclustering_noise_loss(binary_target,noise_partition,ideal_ratio,emb_vec,alpha, \
    energybins,seq_length,batch_size):
    '''
    Compute the deep clustering loss with modified architecture for noise
    cost function based on Hershey et al. 2016 + cost derivation of ideal ratio mask for noise
    for one mini-batch

    Args:
        binary_target:      optimal binary partitioning of the bins                   [batch_size x time x (feat_dim (F)*nrS)]
        noise_partition:       binary mask indicating the bins where noise dominates     [batch_size x time x feat_dim (F)]
        ideal_ratio:        optimal ratio masker to filter the noise                  [batch_size x time x feat_dim (F)]
        emb_vec:            resulting embeddingsvectors                               [batch_size x time x (feat_dim (F) *emb_dim(D))] tensor containing the embedding vectors
        alpha:              resulting alpha                                           [batch_size x time x feature_dim (F)]
        energybins:         indicating whether the bin has enough energy              [batch_size x time x feature_dim (F)]
        seq_length:         sequence lengths                                          [batch_size]
        batch_size:         number of utterances in this batch
    Returns:
        loss: a scalar value containing the loss
        norm: a scalar value containing the normalisation constant
    '''

    with tf.name_scope('deepclustering_noise_loss'):
        # Number of frequency bins
        F = tf.shape(usedbins)[2]
        # Dimension of embeddingsspace
        D = tf.shape(emb_vec)[2]/F
        # Number of sources
        nrS = tf.shape(binary_target)[2]/F # Number of speakers

        # Initialise loss and normalisation with zero
        loss = 0.0
        norm = 0.0

        # Loop over utterances in this batch
        for utt_ind in range(batch_size):
            # Number of timeframes in this utterance
            T = seq_length[utt_ind]
            # Number of time-frequency bins in current spectogram
            K = T*F

            # Which time-frequency bins have enough enery
            energybins_utt = energybins[utt_ind]
            energybins_utt = energybins_utt[:T,:] # dim: (T x F)
            # Embedding vectors of this utterance
            logits_utt = emb_vec[utt_ind]
            logits_utt = logits_utt[:T,:] # dim: (T x F*D)
            # partition targets for this utterance
            targets_utt = binary_target[utt_ind]
            targets_utt = targets_utt[:T,:] # dim: (T x F*nrS)
            # noise partition for current utterance
            noise_partition_utt = noise_partition[utt_ind]
            noise_partition_utt = noise_partition_utt[:T,:] # dim: (T x F)
            # ideal ratio for current utterance
            ideal_ratio_utt = ideal_ratio[utt_ind]
            ideal_ratio_utt = ideal_ratio_utt[:T,:] # dim: (T x F)
            # Alpha outputs of the network
            alpha_utt = alpha[utt_ind]
            alpha_utt = alpha_utt[:T,:] # dim: (TxF)


            # Reshape the energy indication
            ebresh=tf.reshape(energybins_utt,[K,1],name='ubresh')
            ebreshV=tf.tile(ebresh,[1,D])
            ebreshV=tf.to_float(ebreshV) # dim: (K x D)
            ebreshY=tf.tile(ebresh,[1,nrS])  # dim: (K x nrS)

            # Reshape binary noise mask
            nbresh = tf.reshape(1-noise_partition_utt,[K,1],name='ndresh')
            nbreshV = tf.tile(nbresh,[1,D])
            nbreshV = tf.to_float(nbreshV) # dim: (K x D)
            nbreshY = tf.tile(nbresh,[1,nrS]) # dim: (K x nrS)

            # Reshape the embeddingsvectors
            V=tf.reshape(logits_utt,[K,D],name='V') # dim: (K x D)
            # Normalise Embeddingsvectors
            Vnorm=tf.nn.l2_normalize(V, axis=1, epsilon=1e-12, name='Vnorm')
            # Remove noisy and silent bins
            Vnorm=tf.multiply(tf.multiply(Vnorm,ubreshV),nbreshV) # dim: (K x D)
            # Reshape binary targets and remove noisy and silent bins
            Y=tf.reshape(targets_utt,[K,nrS],name='Y')
            Y=tf.multiply(tf.multiply(Y,ubreshY),ndreshY)
            Y=tf.to_float(Y) # dim: (K x nrS)

            # First part cost function cfr. Hershey et al. 2016
            prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, a_is_sparse=True,
	                b_is_sparse=True, name='VTV')
            prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, a_is_sparse=True,
	                b_is_sparse=True, name='VTY')

            term1=tf.reduce_sum(tf.square(prod1),name='frob_1')
            term2=tf.reduce_sum(tf.square(prod2),name='frob_2')

            loss_utt_1 = tf.add(term1,-2*term2,name='term1and2')
            # Number of non noisy non silence cells squared
            norm_1= tf.square(tf.to_float(tf.reduce_sum(tf.multiply(nbresh,ubresh))))
            norm_1 = tf.maximum(norm_1,1)

            # Ideal ratio masker reshape
            ideal_ratio_utt_resh = tf.reshape(ideal_ratio_utt,[K,1],name='ideal_ratio')
            # Alpha output network
            noise_actual = tf.reshape(alpha_utt,[K,1],name='nactual')
            loss_utt_2 = tf.reduce_sum(tf.square(ideal_ratio_utt_resh-noise_actual))

            # Normalisation constant: number of cells in spectrogram
            norm_2 = tf.to_float(Nspec)
            loss += loss_utt_1/norm_1 + loss_utt_2/norm_2


    norm = tf.to_float(tf.constant(batch_size))
    return loss , norm

def pit_loss(targets, logits, mix_to_mask, seq_length, batch_size):
    '''
    Compute the permutation invariant loss.
    Remark: There is actually a more efficient approach to calculate this loss. First calculate
    the loss for every reconstruction to every target ==> nrS^2 combinations and then add
    together the losses to form every possible permutation.

    Args:
        targets: a [batch_size x time x feat_dim  x nrS] tensor containing the multiple targets
        logits: a [batch_size x time x (feat_dim*nrS)] tensor containing the logits
        mix_to_mask: a [batch_size x time x feat_dim] tensor containing the mixture that will be masked
        seq_length: a [batch_size] vector containing the
            sequence lengths
        batch_size: the batch size

    Returns:
        a scalar value containing the loss
    '''


    with tf.name_scope('PIT_loss'):
        feat_dim = tf.shape(targets)[2]

        output_dim = tf.shape(logits)[2]
        nrS = targets.get_shape()[3]
        nrS_tf = tf.shape(targets)[3]
        permutations = list(itertools.permutations(range(nrS),nrS))

        loss = 0.0
        norm = tf.to_float(nrS_tf * feat_dim * tf.reduce_sum(seq_length))
        for utt_ind in range(batch_size):
	    N = seq_length[utt_ind]
	    logits_utt = logits[utt_ind]
	    logits_utt = logits_utt[:N,:]
	    targets_utt = targets[utt_ind]
	    targets_utt = targets_utt[:N,:,:]
	    mix_to_mask_utt = mix_to_mask[utt_ind]
	    mix_to_mask_utt = mix_to_mask_utt[:N,:]

	    logits_resh = tf.reshape(logits_utt,[N,feat_dim,nrS_tf])
	    Masks = tf.nn.softmax(logits_resh, dim=2)

	    mix_to_mask_utt = tf.expand_dims(mix_to_mask_utt,-1)
	    recs = tf.multiply(Masks, mix_to_mask_utt)


	    targets_resh = tf.transpose(targets_utt,perm=[2,0,1])

	    recs = tf.transpose(recs,perm=[2,0,1])

	    perm_cost = []
	    for perm in permutations:
		    tmp = tf.square(tf.norm(tf.gather(recs,perm)-targets_resh,ord='fro',axis=[1,2]))
		    perm_cost.append(tf.reduce_sum(tmp))

	    loss_utt = tf.reduce_min(perm_cost)

	    loss += loss_utt

    #loss = loss/tf.to_float(batch_size)

    return loss, norm

def cross_entropy_loss_eos(targets, logits, logit_seq_length,
                           target_seq_length):
    '''
    Compute the cross_entropy loss with an added end of sequence label

    Args:
        targets: a [batch_size x time] tensor containing the targets
        logits: a [batch_size x time x num_classes] tensor containing the logits
        logit_seq_length: a [batch_size] vector containing the
            logit sequence lengths
        target_seq_length: a [batch_size] vector containing the
            target sequence lengths

    Returns:
        a scalar value containing the loss
    '''

    batch_size = tf.shape(targets)[0]

    with tf.name_scope('cross_entropy_loss'):

        output_dim = tf.shape(logits)[2]

        #get the logits for the final timestep
        indices = tf.stack([tf.range(batch_size),
                            logit_seq_length-1],
                           axis=1)
        final_logits = tf.gather_nd(logits, indices)

        #stack all the logits except the final logits
        stacked_logits = seq2nonseq(logits,
                                    logit_seq_length - 1)

        #create the stacked targets
        stacked_targets = seq2nonseq(targets,
                                     target_seq_length)

        #create the targets for the end of sequence labels
        final_targets = tf.tile([output_dim-1], [batch_size])

        #add the final logits and targets
        stacked_logits = tf.concat([stacked_logits, final_logits], 0)
        stacked_targets = tf.concat([stacked_targets, final_targets], 0)

        #compute the cross-entropy loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=stacked_logits,
            labels=stacked_targets)

        loss = tf.reduce_mean(losses)

    return loss

def get_indices(sequence_length):
    '''get the indices corresponding to sequences (and not padding)

    Args:
        sequence_length: the sequence_lengths as a N-D tensor

    Returns:
        A [sum(sequence_length) x N-1] Tensor containing the indices'''

    with tf.name_scope('get_indices'):

        numdims = len(sequence_length.shape)

        #get th emaximal length
        max_length = tf.reduce_max(sequence_length)

        sizes = tf.shape(sequence_length)

        range_tensor = tf.range(max_length-1)
        for i in range(1, numdims):
            tile_dims = [1]*i + [sizes[i]]
            range_tensor = tf.tile(tf.expand_dims(range_tensor, i), tile_dims)

        indices = tf.where(tf.less(range_tensor,
                                   tf.expand_dims(sequence_length, numdims)))

    return indices

def mix(inputs, hidden_dim, scope=None):
    '''mix the layer in the time dimension'''

    with tf.variable_scope(scope or 'mix'):

        #append the possition to the inputs
        position = tf.expand_dims(tf.expand_dims(tf.range(
            tf.shape(inputs)[1]), 0), 2)
        position = tf.cast(position, tf.float32)
        position = tf.tile(position, [tf.shape(inputs)[0], 1, 1])
        expanded_inputs = tf.concat([inputs, position], 2)

        #apply the querry layer
        query = tf.contrib.layers.linear(expanded_inputs, hidden_dim,
                                         scope='query')

        #apply the attention layer
        queried = tf.contrib.layers.linear(expanded_inputs, hidden_dim,
                                           scope='queried')

        #create a sum for every combination of query and attention
        query = tf.expand_dims(query, 0)
        query = tf.tile(query, [tf.shape(query)[2], 1, 1, 1])
        summed = tf.transpose(tf.nn.tanh(query + queried), [1, 2, 0, 3])

        #map the combinations to single values
        attention = tf.contrib.layers.fully_connected(
            inputs=summed,
            num_outputs=1,
            scope='attention',
            activation_fn=tf.nn.tanh
        )[:, :, :, 0]

        #apply softmax to the attention values
        attention = tf.nn.softmax(attention)

        #use the attention to recombine the inputs
        outputs = tf.matmul(attention, inputs)

    return outputs
