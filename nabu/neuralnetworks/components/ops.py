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



def deepattractornet_loss(partition_targets, spectogram_targets, mix_to_mask, usedbins, embeddings, seq_length, batch_size):
    '''
    Compute the deep attractor net loss (as described in Deep attractor network for single-microphone speaker separation,
        Zhuo Chen, et al. [1])

    Args:
        partition_targets: a [batch_size x time (T) x (feature_dim(F)*nr_S)] tensor containing the partition targets (to which class
            belongs a bin)
        spectogram_targets: a [batch_size x time (T) x feature_dim (F)  x nrS] tensor containing
            the clean spectogram of the sources
        mix_to_mask = a [batch_size x time (T) x feature_dim (F)] tensor containing the spectograms of the mixture
        usedbins: a [batch_size x time(T) x feature_dim (F)] tensor indication the bins to use in the loss function calculation
            As suggested in [1] bins with a to low energy are discarted
        embeddings: a [batch_size x time (T) x (feature_dim(F) * emb_dim (K)] tensor containing the embeddingsvectors
        seq_length: a [batch_size] vector containing the sequence lengths
        batch_size: batch_size (# of batches)
    Returns:
        a scalar value containing the loss
    '''
    with tf.name_scope('deepattractornet_loss'):
        # feat_dim : F
        F = tf.shape(usedbins)[2]
        # embedding dimension d
        emb_dim = tf.shape(embeddings)[2]/F
        nr_S= tf.shape(spectogram_targets)[3]

        loss = 0.0
        norm = 0.0

        for batch_ind in range(batch_size):
            # T : length of the current timeframe
            T = seq_length[batch_ind]
            # N: number of bins in current spectogram
            N = T*F
            # Which time/frequency-bins are used in this batch
            usedbins_batch = usedbins[batch_ind]
            usedbins_batch = usedbins_batch[:T,:]
            embedding_batch = embeddings[batch_ind]
            embedding_batch = embedding_batch[:T,:]
            partition_batch = partition_targets[batch_ind]
            partition_batch = partition_batch[:T,:]
            mix_to_mask_batch = mix_to_mask[batch_ind]
            mix_to_mask_batch = mix_to_mask_batch[:T,:]
            spectogram_batch = spectogram_targets[batch_ind]
            spectogram_batch = spectogram_batch[:T,:,:]

            #remove the non_silence (cfr bins above energy thresh) bins. Removing in logits and
    	    #targets will give 0 contribution to loss.
            ubresh = tf.reshape(usedbins_batch,[N,1],name='ubresh')
            #ubreshV= tf.tile(ubresh,[1,emb_dim])
            #ubreshV=tf.to_float(ubreshV)
            ubreshY=tf.tile(ubresh,[1,nr_S])

            # V : matrix containing the embeddingsvectors for this batch,
            # has shape [nb_bins ( =T*F = N ) x emb_dim]
            V = tf.reshape(embedding_batch,[N,emb_dim],name='V')
            # No need to normalize: Vnorm = tf.nn.l2_normalize(V, dim=1, epsilon=1e-12, name='Vnorm')
            # V = tf.multiply(V,ubreshV) # elementwise multiplication
            Y = tf.reshape(partition_batch,[N,nr_S],name='Y')
	   
            Y = tf.multiply(Y,ubreshY)
            Y = tf.to_float(Y)

            numerator_A=tf.matmul(Y,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=True, name='YTV')
            nb_bins_class = tf.reduce_sum(Y,axis = 0) # dim: (rank 1) number_sources 
            nb_bins_class = tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
            nb_bins_class = tf.expand_dims(nb_bins_class,1) # dim: (rank 2) number_sources x 1
            denominator_A = tf.tile(nb_bins_class,[1,emb_dim],name='denominator_A') #number_sources x emb_dim
            A = tf.divide(numerator_A,denominator_A,name='A')

            prod_1 = tf.matmul(A,V,transpose_a=False, transpose_b = True,name='AVT')
            # Softmax als alternatief?? Nakijken paper +testen
            M = tf.sigmoid(prod_1,name='M') # dim: number_sources x N
             # eliminate nan introduced by no dominant bins of speaker

            X = tf.transpose(tf.reshape(mix_to_mask_batch,[N,1],name='X'))
            masked_sources = tf.multiply(M,X) # dim: number_sources x N
            S = tf.reshape(tf.transpose(spectogram_batch,perm=[2,0,1]),[nr_S,N])
            loss_utt = tf.reduce_sum(tf.square(S-masked_sources),name='loss')
            norm += tf.to_float(tf.reduce_sum(ubresh))
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

def deepclustering_noise_loss(speech_target,noise_target, emb_vec,noise_detect_output, usedbins,\
			        seq_length,batch_size):
    '''
    Compute the deep clustering loss
    cost function based on Hershey et al. 2016

    Args:
        speech_target: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
        noise_target: a [batch_size x time x feat_dim] tensor containing the whether the bin is dominated by noise
        emb_vec: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the embedding vectors
        noise_detect_output: a [batch_size x time x feat_dim] containing outputs for noise detection
        usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
        seq_length: a [batch_size] vector containing the sequence lengths
        batch_size: the batch size

    Returns:
        a scalar value containing the loss
    '''

    with tf.name_scope('deepclustering_noise_loss'):
        F = tf.shape(usedbins)[2]
        emb_dim = tf.shape(emb_vec)[2]/F
        target_dim = tf.shape(speech_target)[2]
        nrS = target_dim/F
        loss = 0.0
        norm = 0.0

        for utt_ind in range(batch_size):
            T = seq_length[utt_ind]
            Nspec = T*F
            usedbins_utt = usedbins[utt_ind]
            usedbins_utt = usedbins_utt[:T,:]
            
            logits_utt = emb_vec[utt_ind]
            logits_utt = logits_utt[:T,:]
            targets_utt = speech_target[utt_ind]
            targets_utt = targets_utt[:T,:]
            noise_target_utt = noise_target[utt_ind]
            noise_target_utt = noise_target_utt[:T,:]
            noise_detect_output_utt = noise_detect_output[utt_ind]
            noise_detect_output_utt = noise_detect_output_utt[:T,:]
            

            #remove the non_silence (cfr bins below energy thresh) bins. Removing in logits and
            #targets will give 0 contribution to loss.
            ubresh=tf.reshape(usedbins_utt,[Nspec,1],name='ubresh')
            ubreshV=tf.tile(ubresh,[1,emb_dim])
            ubreshV=tf.to_float(ubreshV)
            ubreshY=tf.tile(ubresh,[1,nrS])
            
            # TODO nakijken wat type ndresh is
            ndresh = tf.reshape(1-noise_target_utt,[Nspec,1],name='ndresh')
            
            ndreshV = tf.tile(ndresh,[1,emb_dim])
            ndreshV = tf.to_float(ndreshV)
            ndreshY = tf.tile(ndresh,[1,nrS])
            
	    
            V=tf.reshape(logits_utt,[Nspec,emb_dim],name='V')
            Vnorm=tf.nn.l2_normalize(V, dim=1, epsilon=1e-12, name='Vnorm')
            Vnorm=tf.multiply(tf.multiply(Vnorm,ubreshV),ndreshV)
            Y=tf.reshape(targets_utt,[Nspec,nrS],name='Y')
            Y=tf.multiply(tf.multiply(Y,ubreshY),ndreshY)
            Y=tf.to_float(Y)

            prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, a_is_sparse=True,
	                b_is_sparse=True, name='VTV')
            prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, a_is_sparse=True,
	                b_is_sparse=True, name='VTY')

            term1=tf.reduce_sum(tf.square(prod1),name='frob_1')
            term2=tf.reduce_sum(tf.square(prod2),name='frob_2')

            loss_utt_1 = tf.add(term1,-2*term2,name='term1and2')
            norm_1= tf.square(tf.to_float(tf.reduce_sum(tf.multiply(1-noise_target_utt,usedbins_utt))))
            norm_1 = tf.maximum(norm_1,1)
 
            noise_desired = tf.to_float(tf.reshape(noise_target_utt,[Nspec,1],name='ndesired'))
            noise_actual = tf.reshape(noise_detect_output_utt,[Nspec,1],name='nactual')
            loss_utt_2 = tf.reduce_sum(tf.square(noise_desired-noise_actual))
            
            norm_2 = tf.to_float(Nspec)
            #normalizer= tf.to_float(tf.square(tf.reduce_sum(ubresh)))
            #loss += loss_utt/normalizer*(10**9)
            loss += loss_utt_1/norm_1 + loss_utt_2/norm_2


    #loss = loss/tf.to_float(batch_size)

    return loss , tf.constant(1.)
    
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
