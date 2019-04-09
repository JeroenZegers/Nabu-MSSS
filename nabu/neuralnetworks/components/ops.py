"""@file ops.py
some operations"""

import tensorflow as tf
import itertools
import math
from tensorflow.python.framework import ops
import pdb


def unit_activation(x, name=None):

	with ops.name_scope(name, "unit_activation", [x]) as name:
		x = ops.convert_to_tensor(x, name="x")
		return tf.ones(tf.shape(x))


def squash(s, axis=-1, epsilon=1e-7, name=None):
	"""squash function"""
	with tf.name_scope(name, default_name="squash"):
		squared_norm=tf.reduce_sum(tf.square(s), axis=axis,
									 keepdims=True)
		sn=tf.sqrt(squared_norm + epsilon)
		squash_factor=squared_norm / (1. + squared_norm)
		squash_factor /= sn

		return squash_factor * s


def safe_norm(s, axis=-1, keepdims=False, epsilon=1e-7, name=None):
	"""compute a safe norm"""

	with tf.name_scope(name, default_name='safe_norm'):
		x=tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
		return tf.sqrt(x + epsilon)


class VoteTranformInitializer(tf.keras.initializers.Initializer):
	"""An Initializer for the voting transormation matrix in a capsule layer"""

	def __init__(self,
				 scale=1.0,
				 mode='fan_in',
				 distribution='normal',
				 seed=None,
				 dtype=tf.float32):
		"""
		Constructor
		args:
			scale: how to scale the initial values (default: 1.0)
			mode: One of 'fan_in', 'fan_out', 'fan_avg'. (default: fan_in)
			distribution: one of 'uniform' or 'normal' (default: normal)
			seed: A Python integer. Used to create random seeds.
			dtype: The data type. Only floating point types are supported.
		"""

		if scale <= 0.:
			raise ValueError('`scale` must be positive float.')
		if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
			raise ValueError('Invalid `mode` argument:', mode)
		distribution = distribution.lower()
		if distribution not in {'normal', 'uniform'}:
			raise ValueError('Invalid `distribution` argument:', distribution)


		self.scale = scale
		self.mode = mode
		self.distribution = distribution
		self.seed = seed
		self.dtype = tf.as_dtype(dtype)

	def __call__(self, shape, dtype=None, partition_info=None):
		"""initialize the variables
		args:
			shape: List of `int` representing the shape of the output `Tensor`.
				[num_capsules_in, capsule_dim_in, num_capsules_out,
				 capsule_dim_out]
			dtype: (Optional) Type of the output `Tensor`.
			partition_info: (Optional) variable_scope._PartitionInfo object
				holding additional information about how the variable is
				partitioned. May be `None` if the variable is not partitioned.
		Returns:
			A `Tensor` of type `dtype` and `shape`.
		"""

		if dtype is None:
			dtype = self.dtype
		scale = self.scale
		scale_shape = shape

		if partition_info is not None:
			scale_shape = partition_info.full_shape

		if len(scale_shape) != 4:
			raise ValueError('expected shape to be of length 4', scale_shape)

		fan_in = scale_shape[1]
		fan_out = scale_shape[3]

		if self.mode == 'fan_in':
			scale /= max(1., fan_in)
		elif self.mode == 'fan_out':
			scale /= max(1., fan_out)
		else:
			scale /= max(1., (fan_in + fan_out) / 2.)

		if self.distribution == 'normal':
			stddev = math.sqrt(scale)
			return tf.truncated_normal(
				shape, 0.0, stddev, dtype, seed=self.seed)
		else:
			limit = math.sqrt(3.0 * scale)
			return tf.random_uniform(
				shape, -limit, limit, dtype, seed=self.seed)

	def get_config(self):
		"""get the initializer config"""

		return {
			'scale': self.scale,
			'mode': self.mode,
			'distribution': self.distribution,
			'seed': self.seed,
			'dtype': self.dtype.name
		}

def capsule_initializer(scale=1.0, seed=None, dtype=tf.float32):
	"""a VoteTranformInitializer"""

	return VoteTranformInitializer(
		scale=scale,
		mode='fan_avg',
		distribution='uniform',
		seed=seed,
		dtype=dtype
	)

def seq2nonseq(sequential, sequence_lengths, name=None):
	"""
	Convert sequential data to non sequential data

	Args:
		sequential: the sequential data which is a [batch_size, max_length, dim]
			tensor
		sequence_lengths: a [batch_size] vector containing the sequence lengths
		name: [optional] the name of the operation

	Returns:
		non sequential data, which is a TxF tensor where T is the sum of all
		sequence lengths
	"""

	with tf.name_scope(name or 'seq2nonseq'):

		indices = get_indices(sequence_lengths)

		# create the values
		tensor = tf.gather_nd(sequential, indices)


	return tensor

def dense_sequence_to_sparse(sequences, sequence_lengths):
	"""convert sequence dense representations to sparse representations

	Args:
		sequences: the dense sequences as a [batch_size x max_length] tensor
		sequence_lengths: the sequence lengths as a [batch_size] vector

	Returns:
		the sparse tensor representation of the sequences
	"""

	with tf.name_scope('dense_sequence_to_sparse'):

		# get all the non padding sequences
		indices = tf.cast(get_indices(sequence_lengths), tf.int64)

		# create the values
		values = tf.gather_nd(sequences, indices)

		# the shape
		shape = tf.cast(tf.shape(sequences), tf.int64)

		sparse = tf.SparseTensor(indices, values, shape)

	return sparse

def deepattractornet_sigmoid_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings, \
	seq_length, batch_size):
	"""
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
	"""
	with tf.name_scope('deepattractornet_sigmoid_loss'):
		# Number of frequency bins
		F = tf.shape(energybins)[2]
		# Dimension embeddingspace
		D = tf.shape(embeddings)[2]/F
		# Number of sources
		nrS = tf.shape(spectogram_targets)[3]

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
			ebreshY = tf.tile(ebresh,[1,nrS]) # dim: (K x nrS)

			# Reshape the embeddingsvectors
			V = tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

			# Reshape partitioning and remove silent bins
			Y = tf.reshape(partition_utt,[K,nrS],name='Y')
			Y = tf.multiply(Y,ebreshY)
			Y = tf.to_float(Y) # dim: (K x nrS)

			# Calculate attractor centers
			numerator_A = tf.matmul(Y,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
			# Number of bins each speaker dominates
			nb_bins_class = tf.reduce_sum(Y,axis=0) # dim: (rank 1) (nrS)
			# Set number of bins of each speaker to at least 1 to avoid division by zero
			nb_bins_class = tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
			nb_bins_class = tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
			denominator_A = tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
			A = tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

			# Calculate sigmoid mask
			prod_1 = tf.matmul(A,V,transpose_a=False, transpose_b=True,name='AVT')
			M = tf.sigmoid(prod_1,name='M') # dim: (nrS x K)

			# Reshape spectrogram of mixture
			X = tf.transpose(tf.reshape(mix_to_mask_utt,[K,1],name='X')) # dim: (1 x K)
			# Calculate the reconstructions
			reconstructions = tf.multiply(M,X) # dim: (nrS x K)
			S=tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
			loss_utt=tf.reduce_sum(tf.square(S-reconstructions),name='loss')
			# update loss and normalisation
			norm += tf.to_float(K*nrS)
			loss += loss_utt

		return loss,norm


def deepattractornet_softmax_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings,\
				seq_length, batch_size):
	"""
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
	"""
	with tf.name_scope('deepattractornet_softmax_loss'):
		# Number of frequency bins
		F=tf.shape(energybins)[2]
		# Dimension embeddingspace
		D=tf.shape(embeddings)[2]/F
		# Number of sources
		nrS= tf.shape(spectogram_targets)[3]

		# Initialise loss and normalisation with zero
		loss=0.0
		norm=0.0

		# Loop over utterances in this batch
		for utt_ind in range(batch_size):
			# Number of timeframes in this utterance
			T=seq_length[utt_ind]
			# Number of time-frequency bins in current spectogram
			K=T*F
			# Which time-frequency bins have enough enery
			energybins_utt=energybins[utt_ind]
			energybins_utt=energybins_utt[:T,:] # dim: (Tx F)
			# Embedding vectors of this utterance
			embedding_utt=embeddings[utt_ind]
			embedding_utt=embedding_utt[:T,:] # dim: (T x F*D)
			# Partitioning of this utterance
			partition_utt=partitioning[utt_ind]
			partition_utt=partition_utt[:T,:] # dim: (T x F*nrS)
			# Spectrogram of this utterance
			mix_to_mask_utt=mix_to_mask[utt_ind]
			mix_to_mask_utt=mix_to_mask_utt[:T,:] # dim: (T x F)
			# Clean spectogram of the speakers in this utterance
			spectogram_utt=spectogram_targets[utt_ind]
			spectogram_utt=spectogram_utt[:T,:,:] # dim: (T x F x nrS)

			# Reshape the energy indication
			ebresh=tf.reshape(energybins_utt,[K,1],name='ebresh') # dim: (K x 1)
			ebreshY=tf.tile(ebresh,[1,nrS]) # dim: (K x nrS)

			# Reshape the embeddingsvectors
			V=tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

			# Reshape partitioning and remove silent bins
			Y=tf.reshape(partition_utt,[K,nrS],name='Y') # dim: N x number_sources
			Y=tf.multiply(Y,ebreshY)
			Y=tf.to_float(Y) # dim: (K x nrS)

			# Calculate attractor centers
			numerator_A=tf.matmul(Y,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
			# Number of bins each speaker dominates
			nb_bins_class=tf.reduce_sum(Y,axis=0) # dim: (rank 1) (nrS)
			# Set number of bins of each speaker to at least 1 to avoid division by zero
			nb_bins_class=tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
			nb_bins_class=tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
			denominator_A=tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
			A=tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

			# Calculate softmax masker
			prod_1=tf.matmul(A,V,transpose_a=False, transpose_b=True,name='AVT')
			M=tf.nn.softmax(prod_1,dim=0,name='M') # dim: (nrS x K)

			# Reshape spectrogram of mixture
			X=tf.transpose(tf.reshape(mix_to_mask_utt,[K,1],name='X')) # dim: (1 x K)
			# Calculate the reconstructions
			reconstructions=tf.multiply(M,X) # dim: (nrS x K)
			S=tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
			# Calculate loss of this utterance
			loss_utt=tf.reduce_sum(tf.square(S-reconstructions),name='loss') # Calculate difference reconstructions and clean spectrogram
			# update loss and normalisation
			norm += tf.to_float(K*nrS)
			loss += loss_utt

		return loss,norm

def deepattractornetnoise_soft_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings, \
		alpha, seq_length, batch_size):
	"""
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

	"""
	with tf.name_scope('deepattractornetnoise_soft_loss'):
		# Number of frequency bins
		F=tf.shape(energybins)[2]
		# Dimension embeddingspace
		D=tf.shape(embeddings)[2]/F
		# Number of sources
		nrS=tf.shape(spectogram_targets)[3]

		# Initialise loss and normalisation with zero
		loss=0.0
		norm=0.0

		# Loop over utterances in this batch
		for utt_ind in range(batch_size):
			# Number of timeframes in this utterance
			T=seq_length[utt_ind]
			# Number of time-frequency bins in current spectogram
			K=T*F
			# Which time-frequency bins have enough enery
			energybins_utt=energybins[utt_ind]
			energybins_utt=energybins_utt[:T,:]  # dim: (T x F)
			# Embedding vectors of this utterance
			embedding_utt=embeddings[utt_ind]
			embedding_utt=embedding_utt[:T,:] # dim: (T x F*D)
			# Partitioning of this utterance
			partition_utt=partitioning[utt_ind]
			partition_utt=partition_utt[:T,:] # dim: (T x F*nrS)
			# Spectrogram of this utterance
			mix_to_mask_utt=mix_to_mask[utt_ind]
			mix_to_mask_utt=mix_to_mask_utt[:T,:] # dim: (T x F)
			# Clean spectogram of the speakers in this utterance
			spectogram_utt=spectogram_targets[utt_ind]
			spectogram_utt=spectogram_utt[:T,:,:] # dim: (T x F x nrS)
			# Alpha outputs of the network
			alpha_utt=alpha[utt_ind]
			alpha_utt=alpha_utt[:T,:] # dim: (TxF)

			# Reshape the energy indication
			ebresh=tf.reshape(energybins_utt,[K,1],name='ubresh') # dim: (K x 1)
			ebreshY=tf.tile(ebresh,[1,nrS],name='ubreshY') # dim: (K x nrS)

			# Value of alpha as weight for bin in calculation attractor
			alpharesh=tf.reshape(alpha_utt,[K,1],name='nbresh') # dim: (K x 1)
			alphareshY=tf.tile(alpharesh,[1,nrS],name='nbreshY') # dim: (K x nrS)

			# Reshape the embeddingsvectors
			V=tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

			# Reshape partitioning, remove silent bins and apply weights
			Y=tf.reshape(partition_utt,[K,nrS],name='Y')
			Y_tilde=tf.to_float(tf.multiply(Y,ebreshY),name='Y_tilde') # dim: (K x nrS)
			Y_tilde_W=tf.multiply(Y_tilde,alphareshY,name='Y_tilde_W') # dim: (K x nrS)

			# Calculate attractor centers
			numerator_A=tf.matmul(Y_tilde_W,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
			nb_bins_class=tf.reduce_sum(Y_tilde_W,axis=0) # dim: (rank 1) (nrS)
			# Set number of bins of each speaker to at least 1 to avoid division by zero
			nb_bins_class=tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
			nb_bins_class=tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
			denominator_A=tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
			A=tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

			# Calculate softmax mask
			prod_1=tf.matmul(A,V,transpose_a=False, transpose_b=True,name='AVT')
			M_speaker=tf.nn.softmax(prod_1,dim=0,name='M') # dim: (nrS x K)

			# Reshape spectrogram of mixture
			X=tf.reshape(mix_to_mask_utt,[K,1],name='X') # dim: (K x 1)
			# Filter noise from mixture
			X_filter_noise=tf.transpose(tf.multiply(X,alpharesh)) # dim: (1 x K)
			reconstructions=tf.multiply(M_speaker,X_filter_noise) # dim: (nrS x K)
			S=tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
			loss_utt=tf.reduce_sum(tf.square(S-reconstructions),name='loss')

			# update loss and normalisation
			norm += tf.to_float(K*nrS)
			loss += loss_utt

		return loss,norm


def deepattractornetnoise_hard_loss(partitioning, spectogram_targets, mix_to_mask, energybins, embeddings, \
			alpha, seq_length, batch_size):
	"""
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
	"""
	with tf.name_scope('deepattractornetnoise_hard_loss'):
		# Number of frequency bins
		F=tf.shape(energybins)[2]
		# Dimension embeddingspace
		D=tf.shape(embeddings)[2]/F
		# Number of sources
		nrS=tf.shape(spectogram_targets)[3]

		# Initialise loss and normalisation with zero
		loss=0.0
		norm=0.0
		# Only time frequency bins with alpha above this threshold are used for calculating the attractorpoints
		noise_threshold=0.75
		# Loop over utterances in this batch
		for utt_ind in range(batch_size):
			# Number of timeframes in this utterance
			T=seq_length[utt_ind]
			# Number of time-frequency bins in current spectogram
			K=T*F
			# Which time-frequency bins have enough enery
			energybins_utt=energybins[utt_ind]
			energybins_utt=energybins_utt[:T,:] # dim: (Tx F)
			# Embedding vectors of this utterance
			embedding_utt=embeddings[utt_ind]
			embedding_utt=embedding_utt[:T,:] # dim: (T x F*D)
			# Partitioning of this utterance
			partition_utt=partitioning[utt_ind]
			partition_utt=partition_utt[:T,:] # dim: (T x F*nrS)
			# Spectrogram of this utterance
			mix_to_mask_utt=mix_to_mask[utt_ind]
			mix_to_mask_utt=mix_to_mask_utt[:T,:] # dim: (T x F)
			# Clean spectogram of the speakers in this utterance
			spectogram_utt=spectogram_targets[utt_ind]
			spectogram_utt=spectogram_utt[:T,:,:] # dim: (T x F x nrS)
			# Alpha outputs of the network
			alpha_utt=alpha[utt_ind]
			alpha_utt=alpha_utt[:T,:] # dim: (TxF)

			# Reshape the energy indication
			ebresh=tf.reshape(energybins_utt,[K,1],name='ubresh')# dim: (K x 1)
			ebreshY=tf.tile(ebresh,[1,nrS]) # dim: (K x nrS)
			# Threshold and reshape alpha
			alpharesh=tf.cast(tf.reshape(alpha_utt,[K,1])>noise_threshold,tf.int32)
			alphareshY=tf.tile(alpharesh,[1,nrS],name='nbreshY') # dim: (K x nrS)

			# Reshape the embeddingsvectors
			V=tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

			# Reshape partitioning, remove silent bins and apply weights
			Y=tf.reshape(partition_utt,[K,nrS],name='Y')
			Y_tilde=tf.to_float(tf.multiply(Y,ebreshY),name='Y_tilde') # dim: (K x nrS)
			Y_tilde_W=tf.multiply(Y_tilde,alphareshY,name='Y_tilde_W') # dim: (K x nrS)

			# Calculate attractor centers
			numerator_A=tf.matmul(Y_tilde_W,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
			nb_bins_class=tf.reduce_sum(Y_tilde_W,axis=0) # dim: (rank 1) (nrS)
			# Set number of bins of each speaker to at least 1 to avoid division by zero
			nb_bins_class=tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
			nb_bins_class=tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
			denominator_A=tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
			A=tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)
			# Calculate softmax mask
			prod_1=tf.matmul(A,V,transpose_a=False, transpose_b=True,name='AVT')

			M_speaker=tf.nn.softmax(prod_1,dim=0,name='M') # dim: (nrS x K)
			M_noise=tf.transpose(tf.reshape(alpha_utt,[K,1])) # dim: (K x 1)

			# Reshape spectrogram of mixture
			X=tf.transpose(tf.reshape(mix_to_mask_utt,[K,1],name='X')) # dim: (1 x K)
			# Filter noise from mixture
			X_filter_noise=tf.multiply(X,M_noise) # dim: (1 x K)
			reconstructions=tf.multiply(M_speaker,X_filter_noise) # dim: number_sources x N
			S=tf.reshape(tf.transpose(spectogram_utt,perm=[2,0,1]),[nrS,K])  # dim: (nrS x K)
			loss_utt=tf.reduce_sum(tf.square(S-reconstructions),name='loss')  # dim: (nrS x K)
			# update loss and normalisation
			norm += tf.to_float(K*nrS)
			loss += loss_utt

		return loss,norm

def noise_filter_loss(clean_spectrogram,noise_spectrogram,alpha,seq_length,batch_size):
	"""
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
	"""
	with tf.name_scope('noise_filter_loss'):
		# Number of frequency bins
		F=tf.shape(noise_spectrogram)[2]
		# Initialise loss and normalisation with zero
		loss=0.0
		norm=0.0
		# Loop over utterances in this batch
		for utt_ind in range(batch_size):
			# Number of timeframes in this utterance
			T=seq_length[utt_ind]
			# Noisy spectrogram
			noise_spectrogram_utt=noise_spectrogram[utt_ind]
			noise_spectrogram_utt=noise_spectrogram_utt[:T,:] # dim: (T x F)
			# Target spectrogram
			clean_spectrogram_utt=clean_spectrogram[utt_ind]
			clean_spectrogram_utt=clean_spectrogram_utt[:T,:] # dim: (T x F)

			# Alpha outputs of the network
			alpha_utt=alpha[utt_ind]
			alpha_utt=alpha_utt[:T,:] # dim: (T x F)

			estimate=tf.multiply(noise_spectrogram_utt,alpha_utt) # dim: (T x F)
			# update loss and normalisation
			loss += tf.reduce_sum(tf.square(clean_spectrogram_utt-estimate),name='loss')
			norm += tf.to_float(T*F)
	return loss,norm

def deepattractornet_noisefilter_loss(partitioning, spectrogram_targets, mix_to_mask, embeddings,alpha,seq_length,
									  batch_size):
	"""
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

	"""
	with tf.name_scope('deepattractornet_noisefilter'):
		# Number of frequency bins
		F=tf.shape(mix_to_mask)[2]
		# Dimension embeddingspace
		D=tf.shape(embeddings)[2]/F
		# Number of sources
		nrS= tf.shape(spectrogram_targets)[3]

		# Initialise loss and normalisation with zero
		loss=0.0
		norm=0.0
		# Threshold on energy
		usedbin_threshold=100.
		# Loop over utterances in this batch
		for utt_ind in range(batch_size):
			# Number of timeframes in this utterance
			T=seq_length[utt_ind]
			# Number of time-frequency bins in current spectogram
			K=T*F
			# Embedding vectors of this utterance
			embedding_utt=embeddings[utt_ind]
			embedding_utt=embedding_utt[:T,:] # dim: (T x F*D)
			# Partitioning of this utterance
			partition_utt=partitioning[utt_ind]
			partition_utt=partition_utt[:T,:] # dim: (T x F*nrS)
			# Spectrogram of this utterance
			mix_to_mask_utt=mix_to_mask[utt_ind]
			mix_to_mask_utt=mix_to_mask_utt[:T,:] # dim: (T x F)
			# Clean spectogram of the speakers in this utterance
			spectrogram_utt=spectrogram_targets[utt_ind]
			spectrogram_utt=spectrogram_utt[:T,:,:] # dim: (T x F x nrS)

			alpha_utt=alpha[utt_ind]
			alpha_utt=alpha_utt[:T,:] # dim: (TxF)

			# Eliminate noise noise
			X_hat_clean=tf.reshape(tf.multiply(mix_to_mask_utt,alpha_utt),[K,1]) # dim: (T x F)

			# Calculate bins with enough energy in cleaned spectrogram
			maxbin=tf.reduce_max(X_hat_clean)
			floor=maxbin/usedbin_threshold
			ubresh=tf.cast(tf.greater(X_hat_clean,floor),tf.int32)
			ubreshY=tf.tile(ubresh,[1,nrS]) # dim: (K x nrS)

			# Reshape the embeddingsvectors
			V=tf.reshape(embedding_utt,[K,D],name='V') # dim: (K x D)

			# Reshape partitioning and remove silent bins
			Y=tf.reshape(partition_utt,[K,nrS],name='Y')
			Y_tilde=tf.multiply(Y,ubreshY)
			Y_tilde=tf.to_float(Y_tilde)

			# Calculate attractor centers
			numerator_A=tf.matmul(Y_tilde,V,transpose_a=True, transpose_b=False, a_is_sparse=True,b_is_sparse=False, name='YTV')
			nb_bins_class=tf.reduce_sum(Y_tilde,axis=0) # dim: (rank 1) (nrS)
			# Set number of bins of each speaker to at least 1 to avoid division by zero
			nb_bins_class=tf.where(tf.equal(nb_bins_class,tf.zeros_like(nb_bins_class)), tf.ones_like(nb_bins_class), nb_bins_class)
			nb_bins_class=tf.expand_dims(nb_bins_class,1) # dim: (rank 2) (nrS x 1)
			denominator_A=tf.tile(nb_bins_class,[1,D],name='denominator_A') # dim: (nrS x D)
			A=tf.divide(numerator_A,denominator_A,name='A') # dim: (nrS x D)

			# Calculate softmax mask
			prod_1=tf.matmul(A,V,transpose_a=False, transpose_b=True,name='AVT')
			M_speaker=tf.nn.softmax(prod_1,dim=0,name='M') # dim: number_sources x N
			# Calculate reconstructions
			reconstructions=tf.multiply(M_speaker,tf.transpose(X_hat_clean)) # dim: (nrS x K)
			S=tf.reshape(tf.transpose(spectrogram_utt,perm=[2,0,1]),[nrS,K]) # dim: (nrS x K)
			loss_utt=tf.reduce_sum(tf.square(S-reconstructions),name='loss')
			# update loss and normalisation
			norm += tf.to_float(K*nrS)
			loss += loss_utt

		return loss,norm

def L41_loss(targets, bin_embeddings, spk_embeddings, usedbins, seq_length, batch_size):
	"""
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
	"""

	with tf.name_scope('L41_loss'):
		feat_dim=tf.shape(usedbins)[2]
		output_dim=tf.shape(bin_embeddings)[2]
		emb_dim=output_dim/feat_dim
		target_dim=tf.shape(targets)[2]
		nrS=target_dim/feat_dim

		loss=0.0
		norm=0

		for utt_ind in range(batch_size):
			N=seq_length[utt_ind]
			usedbins_utt=usedbins[utt_ind]
			usedbins_utt=usedbins_utt[:N,:]
			bin_emb_utt=bin_embeddings[utt_ind]
			bin_emb_utt=bin_emb_utt[:N,:]
			targets_utt=targets[utt_ind]
			targets_utt=targets_utt[:N,:]
			spk_emb_utt=spk_embeddings[utt_ind]

			vi=tf.reshape(bin_emb_utt,[N,feat_dim,1,emb_dim],name='vi')
			vi_norm=tf.nn.l2_normalize(vi,3,name='vi_norm')
			vo=tf.reshape(spk_emb_utt,[1,1,nrS,emb_dim],name='vo')
			vo_norm=tf.nn.l2_normalize(vo,3,name='vo_norm')

			dot=tf.reduce_sum(vi_norm*vo_norm,3,name='D')

			Y=tf.to_float(tf.reshape(targets_utt,[N,feat_dim,nrS]))
			Y=(Y-0.5)*2.0

			# Compute the cost for every element
			loss_utt=-tf.log(tf.nn.sigmoid(Y * dot))

			loss_utt=tf.reduce_sum(tf.to_float(tf.expand_dims(usedbins_utt,-1))*loss_utt)

			loss += loss_utt

			norm += tf.to_float(tf.reduce_sum(usedbins_utt)*nrS)

	# loss=loss/tf.to_float(batch_size)

	return loss , norm

def pit_L41_loss(targets, bin_embeddings, spk_embeddings, mix_to_mask, seq_length, batch_size):
	"""
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
	"""

	with tf.name_scope('PIT_L41_loss'):
		feat_dim=tf.shape(targets)[2]
		output_dim=tf.shape(bin_embeddings)[2]
		emb_dim=output_dim/feat_dim
		target_dim=tf.shape(targets)[2]
		nrS=targets.get_shape()[3]
		nrS_tf=tf.shape(targets)[3]
		permutations=list(itertools.permutations(range(nrS),nrS))

		loss=0.0
		norm=tf.to_float(nrS_tf * feat_dim * tf.reduce_sum(seq_length))

		for utt_ind in range(batch_size):
			N=seq_length[utt_ind]
			bin_emb_utt=bin_embeddings[utt_ind]
			bin_emb_utt=bin_emb_utt[:N,:]
			targets_utt=targets[utt_ind]
			targets_utt=targets_utt[:N,:,:]
			spk_emb_utt=spk_embeddings[utt_ind]
			mix_to_mask_utt=mix_to_mask[utt_ind]
			mix_to_mask_utt=mix_to_mask_utt[:N,:]

			vi=tf.reshape(bin_emb_utt,[N,feat_dim,1,emb_dim],name='vi')
			vi_norm=tf.nn.l2_normalize(vi,3,name='vi_norm')
			vo=tf.reshape(spk_emb_utt,[1,1,nrS_tf,emb_dim],name='vo')
			vo_norm=tf.nn.l2_normalize(vo,3,name='vo_norm')

			D=tf.divide(1,tf.norm(tf.subtract(vi_norm,vo_norm),ord=2,axis=3))
			Masks=tf.nn.softmax(D, axis=2)

			# The masks are estimated, the remainder is the same as in pit_loss
			mix_to_mask_utt=tf.expand_dims(mix_to_mask_utt,-1)
			recs=tf.multiply(Masks, mix_to_mask_utt)

			targets_resh=tf.transpose(targets_utt,perm=[2,0,1])
			recs=tf.transpose(recs,perm=[2,0,1])

			perm_cost=[]
			for perm in permutations:
				tmp=tf.square(tf.norm(tf.gather(recs,perm)-targets_resh,ord='fro',axis=[1,2]))
				perm_cost.append(tf.reduce_sum(tmp))

			loss_utt=tf.reduce_min(perm_cost)

			loss += loss_utt

	# loss=loss/tf.to_float(batch_size)

	return loss , norm

def intravar2centervar_rat_loss(targets, logits, usedbins, seq_length, batch_size):
	"""
	Not realy LDA. numerator is same as above (mean intra class variance), the denominator is the
	variance between the class means (e.g. for 2 classes this equals to the square of halve the distance
	between the 2 means)

	Args:
		targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size

	Returns:
		a scalar value containing the loss
	"""
	print 'Using intravar2centervar_rat_loss'
	with tf.name_scope('intravar2centervar_rat_loss'):
		feat_dim=tf.shape(usedbins)[2]
		output_dim=tf.shape(logits)[2]
		emb_dim=output_dim/feat_dim
		target_dim=tf.shape(targets)[2]
		nrS=target_dim/feat_dim

		loss=0.0
		norm=tf.constant(0.0)

		for utt_ind in range(batch_size):
			N=seq_length[utt_ind]
			Nspec=N*feat_dim
			usedbins_utt=usedbins[utt_ind]
			usedbins_utt=usedbins_utt[:N,:]
			logits_utt=logits[utt_ind]
			logits_utt=logits_utt[:N,:]
			targets_utt=targets[utt_ind]
			targets_utt=targets_utt[:N,:]

			ubresh=tf.cast(tf.reshape(usedbins_utt,[Nspec]),tf.bool,name='ubresh')

			V=tf.reshape(logits_utt,[Nspec,emb_dim])
			V=tf.boolean_mask(V,ubresh,name='V')
			Vnorm=tf.nn.l2_normalize(V, axis=1, epsilon=1e-12, name='Vnorm')
			Y=tf.reshape(targets_utt,[Nspec,nrS])
			Y=tf.boolean_mask(Y,ubresh,name='Y')
			Y=tf.to_float(Y)

			YTY=tf.matmul(Y,Y,transpose_a=True)
			Ycnt=tf.diag_part(YTY)
			Ycnt=tf.expand_dims(Ycnt,-1)+1e-12
			sum_s=tf.matmul(Y,Vnorm,transpose_a=True)
			mean_s=tf.divide(sum_s,Ycnt)
			VminYmean_S=Vnorm-tf.matmul(Y,mean_s)
			dev=tf.reduce_sum(tf.square(VminYmean_S),1,keep_dims=True)
			sum_dev_s=tf.matmul(Y,dev,transpose_a=True)
			mean_dev_s=tf.divide(sum_dev_s,Ycnt)
			intra_cluster_variance=tf.reduce_mean(mean_dev_s)

			_,inter_mean_var=tf.nn.moments(mean_s,0)
			inter_mean_var=tf.reduce_sum(inter_mean_var)+1e-12

			# if only 1 sample in a cluster, just return 1.0
			loss_utt=tf.cond(tf.reduce_min(Ycnt) > 1.1, lambda:
				  tf.divide(intra_cluster_variance,inter_mean_var), lambda: tf.constant(1.0))

			loss += loss_utt

			norm += 1.0

	return loss , norm

def dist2mean_epsilon_closest_rat_loss(targets, logits, usedbins, seq_length, batch_size,rat_power=1,
			   fracbins=None,epsilon=0.2):
	"""
	Not realy LDA. For each embedding determine the ratio of distance to its class center to distance to
	other closest class center

	Args:
		targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size
		rat_power: the loss is the ratio to the power of rat_power
		fracbins: a [batch_size x time x feat_dim] tensor, similar to usedbins, but allowing
		for a gradual assigment between 0 and 1

	Returns:
		a scalar value containing the loss
	"""
	print 'Using dist2mean_closest_rat_loss'
	with tf.name_scope('dist2mean_closest_rat_loss'):
		feat_dim=usedbins.get_shape()[2]
		output_dim=logits.get_shape()[2]
		emb_dim=output_dim/feat_dim
		target_dim=targets.get_shape()[2]
		nrS=target_dim/feat_dim


		ubresh=tf.to_float(tf.reshape(usedbins,[batch_size,-1],name='ubresh') )
		ubresh_expand=tf.expand_dims(ubresh,-1)
		V=tf.reshape(logits,[batch_size,-1,emb_dim])
		V=tf.multiply(V,ubresh_expand,name='V')
		Vnorm=tf.nn.l2_normalize(V, axis=2, epsilon=1e-12, name='Vnorm')
		Y=tf.reshape(targets,[batch_size,-1,nrS])
		Y=tf.to_float(Y)
		Y=tf.multiply(Y,ubresh_expand,name='Y')

		Ycnt=tf.expand_dims(tf.reduce_sum(Y,1),-1)+1e-12
		sum_s=tf.matmul(tf.transpose(Y,[0,2,1]),Vnorm)
		mean_s=tf.divide(sum_s,Ycnt)
		mean_s_resh=tf.expand_dims(tf.transpose(mean_s,[0,2,1]),1)
		Vnorm_resh=tf.expand_dims(Vnorm,-1)
		dev=tf.reduce_sum(tf.square(Vnorm_resh-mean_s_resh),2)
		rat=tf.reduce_sum(dev*Y,2)/(tf.reduce_min(dev+Y*1e20,2)+epsilon)
		rat=rat*ubresh

		if rat_power==2:
			rat=tf.square(rat)
		elif rat_power!=1:
			rat=rat**rat_power

		if fracbins!=None:
			fracbins_resh=tf.reshape(fracbins,[batch_size, -1])
			fracbins_act=tf.multiply(fracbins_resh,ubresh)
			rat*=fracbins_act

		loss= tf.reduce_sum(rat)

		if fracbins==None:
			norm=tf.to_float(tf.reduce_sum(usedbins))
		else:
			norm=tf.reduce_sum(fracbins_act)

	return loss , norm

def dist2mean_closest_rat_loss(targets, logits, usedbins, seq_length, batch_size,rat_power=1,
			   fracbins=None):
	"""
	Not realy LDA. For each embedding determine the ratio of distance to its class center to distance to
	other closest class center

	Args:
		targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size
		rat_power: the loss is the ratio to the power of rat_power
		fracbins: a [batch_size x time x feat_dim] tensor, similar to usedbins, but allowing
		for a gradual assigment between 0 and 1

	Returns:
		a scalar value containing the loss
	"""
	print 'Using dist2mean_closest_rat_loss'
	with tf.name_scope('dist2mean_closest_rat_loss'):
		feat_dim=usedbins.get_shape()[2]
		output_dim=logits.get_shape()[2]
		emb_dim=output_dim/feat_dim
		target_dim=targets.get_shape()[2]
		nrS=target_dim/feat_dim


		ubresh=tf.to_float(tf.reshape(usedbins,[batch_size,-1],name='ubresh') )
		ubresh_expand=tf.expand_dims(ubresh,-1)
		V=tf.reshape(logits,[batch_size,-1,emb_dim])
		V=tf.multiply(V,ubresh_expand,name='V')
		Vnorm=tf.nn.l2_normalize(V, axis=2, epsilon=1e-12, name='Vnorm')
		Y=tf.reshape(targets,[batch_size,-1,nrS])
		Y=tf.to_float(Y)
		Y=tf.multiply(Y,ubresh_expand,name='Y')

		Ycnt=tf.expand_dims(tf.reduce_sum(Y,1),-1)+1e-12
		sum_s=tf.matmul(tf.transpose(Y,[0,2,1]),Vnorm)
		mean_s=tf.divide(sum_s,Ycnt)
		mean_s_resh=tf.expand_dims(tf.transpose(mean_s,[0,2,1]),1)
		Vnorm_resh=tf.expand_dims(Vnorm,-1)
		dev=tf.reduce_sum(tf.square(Vnorm_resh-mean_s_resh),2)
		rat=tf.reduce_sum(dev*Y,2)/(tf.reduce_min(dev+Y*1e20,2)+1.0e-12)
		rat=rat*ubresh

		if rat_power==2:
			rat=tf.square(rat)
		elif rat_power!=1:
			rat=rat**rat_power

		if fracbins!=None:
			fracbins_resh=tf.reshape(fracbins,[batch_size, -1])
			fracbins_act=tf.multiply(fracbins_resh,ubresh)
			rat*=fracbins_act

		loss= tf.reduce_sum(rat)

		if fracbins==None:
			norm=tf.to_float(tf.reduce_sum(usedbins))
		else:
			norm=tf.reduce_sum(fracbins_act)

	return loss , norm

def dist2mean_rat_loss(targets, logits, usedbins, seq_length, batch_size,rat_power=1,
			   fracbins=None):
	"""
	Not realy LDA. For each embedding determine the ratio of distance to its class center to distance to
	other class centers

	Args:
		targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size
		rat_power: the loss is the ratio to the power of rat_power
		fracbins: a [batch_size x time x feat_dim] tensor, similar to usedbins, but allowing
		for a gradual assigment between 0 and 1

	Returns:
		a scalar value containing the loss
	"""
	print 'Using dist2mean_rat_loss'
	with tf.name_scope('dist2mean_rat_loss'):
		feat_dim=usedbins.get_shape()[2]
		output_dim=logits.get_shape()[2]
		emb_dim=output_dim/feat_dim
		target_dim=targets.get_shape()[2]
		nrS=target_dim/feat_dim


		ubresh=tf.to_float(tf.reshape(usedbins,[batch_size,-1],name='ubresh') )
		ubresh_expand=tf.expand_dims(ubresh,-1)
		V=tf.reshape(logits,[batch_size,-1,emb_dim])
		V=tf.multiply(V,ubresh_expand,name='V')
		Vnorm=tf.nn.l2_normalize(V, axis=2, epsilon=1e-12, name='Vnorm')
		Y=tf.reshape(targets,[batch_size,-1,nrS])
		Y=tf.to_float(Y)
		Y=tf.multiply(Y,ubresh_expand,name='Y')

		Ycnt=tf.expand_dims(tf.reduce_sum(Y,1),-1)+1e-12
		sum_s=tf.matmul(tf.transpose(Y,[0,2,1]),Vnorm)
		mean_s=tf.divide(sum_s,Ycnt)
		mean_s_resh=tf.expand_dims(tf.transpose(mean_s,[0,2,1]),1)
		Vnorm_resh=tf.expand_dims(Vnorm,-1)
		dev=tf.reduce_sum(tf.square(Vnorm_resh-mean_s_resh),2)
		rat=tf.reduce_sum(dev*Y,2)/(tf.reduce_sum(dev*(1.0-Y),2)+1e-12)
		rat=rat*ubresh

		if rat_power==2:
			rat=tf.square(rat)
		elif rat_power!=1:
			rat=rat**rat_power

		if fracbins!=None:
			fracbins_resh=tf.reshape(fracbins,[batch_size, -1])
			fracbins_act=tf.multiply(fracbins_resh,ubresh)
			rat*=fracbins_act

		loss= tf.reduce_sum(rat)

		if fracbins==None:
			norm=tf.to_float(tf.reduce_sum(usedbins))
		else:
			norm=tf.reduce_sum(fracbins_act)

	return loss , norm


def deepclustering_loss(targets, logits, usedbins, seq_length, batch_size):
	"""
	Compute the deep clustering loss
	cost function based on Hershey et al. 2016

	Args:
		targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size

	Returns:
		a scalar value containing the loss
	"""

	with tf.name_scope('deepclustering_loss'):
		feat_dim=usedbins.get_shape()[2]
		output_dim=logits.get_shape()[2]
		emb_dim=output_dim/feat_dim
		target_dim=targets.get_shape()[2]
		nrS=target_dim/feat_dim

		ubresh=tf.reshape(usedbins,[batch_size,-1,1],name='ubresh')
		ubresh=tf.to_float(ubresh)

		V=tf.reshape(logits,[batch_size,-1,emb_dim],name='V')
		Vnorm=tf.nn.l2_normalize(V, axis=2, epsilon=1e-12, name='Vnorm')
		Vnorm=tf.multiply(Vnorm,ubresh)
		Y=tf.reshape(targets,[batch_size,-1,nrS],name='Y')
		Y=tf.to_float(Y)
		Y=tf.multiply(Y,ubresh)

		prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, name='VTV')
		prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, name='VTY')
		prod3=tf.matmul(Y,Y,transpose_a=True, transpose_b=False, name='YTY')
		term1=tf.reduce_sum(tf.square(prod1),name='frob_1')
		term2=tf.reduce_sum(tf.square(prod2),name='frob_2')
		term3=tf.reduce_sum(tf.square(prod3),name='frob_3')

		term1and2=tf.add(term1,-2*term2,name='term1and2')
		loss=tf.add(term1and2,term3,name='term1and2and3')
		norm= tf.reduce_sum(tf.square(tf.to_float(tf.reduce_sum(usedbins,[1,2]))))

	return loss , norm

def deepclustering_flat_loss(targets, logits, usedbins, seq_length, batch_size):
	"""
	Compute the deep clustering loss
	cost function based on Hershey et al. 2016

	Args:
		targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits: a [batch_size x time x feat_dim x emb_dim] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size

	Returns:
		a scalar value containing the loss
	"""

	with tf.name_scope('deepclustering_flat_loss'):
		feat_dim=usedbins.get_shape()[2]
		emb_dim=logits.get_shape()[3]
		target_dim=targets.get_shape()[2]
		nrS=target_dim/feat_dim

		ubresh=tf.reshape(usedbins,[batch_size,-1,1],name='ubresh')
		ubresh=tf.to_float(ubresh)

		V=tf.reshape(logits,[batch_size,-1,emb_dim],name='V')
		Vnorm=tf.nn.l2_normalize(V, axis=2, epsilon=1e-12, name='Vnorm')
		Vnorm=tf.multiply(Vnorm,ubresh)
		Y=tf.reshape(targets,[batch_size,-1,nrS],name='Y')
		Y=tf.to_float(Y)
		Y=tf.multiply(Y,ubresh)

		prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, name='VTV')
		prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, name='VTY')
		prod3=tf.matmul(Y,Y,transpose_a=True, transpose_b=False, name='YTY')
		term1=tf.reduce_sum(tf.square(prod1),name='frob_1')
		term2=tf.reduce_sum(tf.square(prod2),name='frob_2')
		term3=tf.reduce_sum(tf.square(prod3),name='frob_3')

		term1and2=tf.add(term1,-2*term2,name='term1and2')
		loss=tf.add(term1and2,term3,name='term1and2and3')
		norm= tf.reduce_sum(tf.square(tf.to_float(tf.reduce_sum(usedbins,[1,2]))))

	return loss , norm

def dc_pit_loss(targets_dc, logits_dc, targets_pit, logits_pit, usedbins, mix_to_mask, seq_length, batch_size,alpha=1.423024812840571e-09):
	"""
	THIS IS OBSOLETE. JUST COMBINE THE 2 LOSSES IN A LOSS COMPUTER
	Compute the joint deep clustering loss and permuation invariant loss
	cost function based on Hershey et al. 2016

	Args:
		targets_dc: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits_dc: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		targets_pit: a [batch_size x time x feat_dim  x nrS)] tensor containing the multiple targets
		logits_pit: a [batch_size x time x (feat_dim*nrS)] tensor containing the logits for pit
		mix_to_mask: a [batch_size x time x feat_dim] tensor containing the mixture that will be masked
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size
		alpha: PIT scaling loss

	Returns:
		a scalar value containing the loss
	"""

	# rougly estimated loss scaling factor so PIT loss and DC loss are more or less of the same magnitude

	with tf.name_scope('dc_pit_loss'):
		feat_dim=usedbins.get_shape()[2]
		output_dc_dim=logits_dc.get_shape()[2]
		emb_dim=output_dc_dim/feat_dim
		target_dc_dim=targets_dc.get_shape()[2]
		output_pit_dim=logits_pit.get_shape()[2]
		nrS=targets_pit.get_shape()[3]
		permutations=list(itertools.permutations(range(nrS),nrS))

		# DC
		ubresh=tf.reshape(usedbins,[batch_size,-1,1],name='ubresh')
		ubresh=tf.to_float(ubresh)

		V=tf.reshape(logits_dc,[batch_size,-1,emb_dim],name='V')
		Vnorm=tf.nn.l2_normalize(V, axis=2, epsilon=1e-12, name='Vnorm')
		Vnorm=tf.multiply(Vnorm,ubresh)
		Y=tf.reshape(targets_dc,[batch_size,-1,nrS],name='Y')
		Y=tf.to_float(Y)
		Y=tf.multiply(Y,ubresh)

		prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, name='VTV')
		prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, name='VTY')
		prod3=tf.matmul(Y,Y,transpose_a=True, transpose_b=False, name='YTY')
		term1=tf.reduce_sum(tf.square(prod1),name='frob_1')
		term2=tf.reduce_sum(tf.square(prod2),name='frob_2')
		term3=tf.reduce_sum(tf.square(prod3),name='frob_3')

		term1and2=tf.add(term1,-2*term2,name='term1and2')
		loss_dc=tf.add(term1and2,term3,name='term1and2and3')
		norm_dc= tf.reduce_sum(tf.square(tf.to_float(tf.reduce_sum(usedbins,[1,2]))))

		# PIT
		logits_pit_resh=tf.transpose(tf.reshape(tf.transpose(logits_pit,[2,0,1]),[nrS,feat_dim,batch_size,-1]),[2,3,1,0])
		Masks=tf.nn.softmax(logits_pit_resh, axis=3)

		mix_to_mask=tf.expand_dims(mix_to_mask,-1)
		recs=tf.multiply(Masks, mix_to_mask)

		targets_pit_resh=tf.transpose(targets_pit,perm=[3,0,1,2])
		recs=tf.transpose(recs,perm=[3,0,1,2])

		perm_cost=[]
		for perm in permutations:
			tmp=tf.square(tf.norm(tf.gather(recs,perm)-targets_pit_resh,ord='fro',axis=[2,3]))
			perm_cost.append(tf.reduce_sum(tmp,0))

		loss_pit_utt=tf.reduce_min(perm_cost,0)
		loss_pit=tf.reduce_sum(loss_pit_utt)
		norm_pit=tf.to_float(tf.reduce_sum(seq_length)*nrS * feat_dim )

		loss=loss_dc/norm_dc+alpha*loss_pit/norm_pit
		norm=tf.constant(1.0)

	return loss , norm


def deepclustering_L1_loss(targets, logits, usedbins, seq_length, batch_size):
	"""
	Compute the deep clustering loss, with L1 norm (instead of frobenius)
	cost function based on Hershey et al. 2016

	Args:
		targets: a [batch_size x time x (feat_dim*nrS)] tensor containing the binary targets
		logits: a [batch_size x time x (feat_dim*emb_dim)] tensor containing the logits
		usedbins: a [batch_size x time x feat_dim] tensor indicating the bins to use in the loss function
		seq_length: a [batch_size] vector containing the sequence lengths
		batch_size: the batch size

	Returns:
		a scalar value containing the loss
	"""

	with tf.name_scope('deepclustering_loss'):
		feat_dim=tf.shape(usedbins)[2]
		output_dim=tf.shape(logits)[2]
		emb_dim=output_dim/feat_dim
		target_dim=tf.shape(targets)[2]
		nrS=target_dim/feat_dim

		loss=0.0
		norm=0.0

		for utt_ind in range(batch_size):
			N=seq_length[utt_ind]
			Nspec=N*feat_dim
			usedbins_utt=usedbins[utt_ind]
			usedbins_utt=usedbins_utt[:N,:]
			logits_utt=logits[utt_ind]
			logits_utt=logits_utt[:N,:]
			targets_utt=targets[utt_ind]
			targets_utt=targets_utt[:N,:]

			# remove the non_silence (cfr bins below energy thresh) bins. Removing in logits and
			# targets will give 0 contribution to loss.
			ubresh=tf.reshape(usedbins_utt,[Nspec,1],name='ubresh')
			ubreshV=tf.tile(ubresh,[1,emb_dim])
			ubreshV=tf.to_float(ubreshV)
			ubreshY=tf.tile(ubresh,[1,nrS])

			V=tf.reshape(logits_utt,[Nspec,emb_dim],name='V')
			Vnorm=tf.nn.l2_normalize(V, axis=1, epsilon=1e-12, name='Vnorm')
			Vnorm=tf.multiply(Vnorm,ubreshV)
			Y=tf.reshape(targets_utt,[Nspec,nrS],name='Y')
			Y=tf.multiply(Y,ubreshY)
			Y=tf.to_float(Y)

			prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, a_is_sparse=True,
					b_is_sparse=True, name='VTV')
			prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, a_is_sparse=True,
					b_is_sparse=True, name='VTY')

			term1=tf.reduce_sum(tf.abs(prod1),name='L1_1')
			term2=tf.reduce_sum(tf.abs(prod2),name='L1_2')

			loss_utt=tf.add(term1,-2*term2,name='term1and2')
			# normalizer= tf.to_float(tf.square(tf.reduce_sum(ubresh)))
			# loss += loss_utt/normalizer*(10**9)
			loss += loss_utt

			norm += tf.square(tf.to_float(tf.reduce_sum(usedbins_utt)))

	# loss=loss/tf.to_float(batch_size)

	return loss , norm

def crossentropy_multi_loss(labels, logits, batch_size):

	with tf.name_scope('crossentropy_multi_loss'):
		nrS=logits.get_shape()[1]
		permutations=list(itertools.permutations(range(nrS),nrS))

		perm_cost=[]
		for perm in permutations:
			logits_resh=tf.gather(logits,perm,axis=1)
			tmp=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits_resh)
			perm_cost.append(tf.reduce_mean(tmp,-1))

		loss= tf.reduce_sum(tf.reduce_min(perm_cost,0))
		norm=tf.to_float(batch_size)

	return loss, norm

def direct_loss(targets, logits, mix_to_mask, seq_length, batch_size):
	"""
	Compute the direct reconstruction loss via masks.

	Args:
		targets: a [batch_size x time x feat_dim  x nrS)] tensor containing the multiple targets
		logits: a [batch_size x time x (feat_dim*nrS)] tensor containing the logits
		mix_to_mask: a [batch_size x time x feat_dim] tensor containing the mixture that will be masked
		seq_length: a [batch_size] vector containing the
			sequence lengths
		batch_size: the batch size

	Returns:
		a scalar value containing the loss
	"""
	with tf.name_scope('direct_loss'):
		feat_dim=targets.get_shape()[2]
		output_dim=logits.get_shape()[2]
		nrS=targets.get_shape()[3]

		logits_resh=tf.transpose(tf.reshape(tf.transpose(logits,[2,0,1]),[nrS,feat_dim,batch_size,-1]),[2,3,1,0])
		Masks=tf.nn.softmax(logits_resh, axis=3)

		mix_to_mask=tf.expand_dims(mix_to_mask,-1)
		recs=tf.multiply(Masks, mix_to_mask)

		norm=tf.to_float(tf.reduce_sum(seq_length)*nrS * feat_dim )

		loss_utt=tf.square(tf.norm(recs-targets,ord='fro',axis=[1,2]))
		loss=tf.reduce_sum(loss_utt)

	return loss, norm

def deepclustering_noise_loss(
		binary_target,noise_partition,ideal_ratio,emb_vec,alpha, energybins,seq_length,batch_size):
	"""
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
	"""

	with tf.name_scope('deepclustering_noise_loss'):
		# Number of frequency bins
		F=tf.shape(energybins)[2]
		# Dimension of embeddingsspace
		D=tf.shape(emb_vec)[2]/F
		# Number of sources
		nrS=tf.shape(binary_target)[2]/F # Number of speakers

		# Initialise loss and normalisation with zero
		loss=0.0
		norm=0.0

		# Loop over utterances in this batch
		for utt_ind in range(batch_size):
			# Number of timeframes in this utterance
			T=seq_length[utt_ind]
			Nspec=T*F
			# Number of time-frequency bins in current spectogram
			K=T*F

			# Which time-frequency bins have enough enery
			energybins_utt=energybins[utt_ind]
			energybins_utt=energybins_utt[:T,:] # dim: (T x F)
			# Embedding vectors of this utterance
			logits_utt=emb_vec[utt_ind]
			logits_utt=logits_utt[:T,:] # dim: (T x F*D)
			# partition targets for this utterance
			targets_utt=binary_target[utt_ind]
			targets_utt=targets_utt[:T,:] # dim: (T x F*nrS)
			# noise partition for current utterance
			noise_partition_utt=noise_partition[utt_ind]
			noise_partition_utt=noise_partition_utt[:T,:] # dim: (T x F)
			# ideal ratio for current utterance
			ideal_ratio_utt=ideal_ratio[utt_ind]
			ideal_ratio_utt=ideal_ratio_utt[:T,:] # dim: (T x F)
			# Alpha outputs of the network
			alpha_utt=alpha[utt_ind]
			alpha_utt=alpha_utt[:T,:] # dim: (TxF)


			# Reshape the energy indication
			ebresh=tf.reshape(energybins_utt,[K,1],name='ubresh')
			ebreshV=tf.tile(ebresh,[1,D])
			ebreshV=tf.to_float(ebreshV) # dim: (K x D)
			ebreshY=tf.tile(ebresh,[1,nrS])  # dim: (K x nrS)

			# Reshape binary noise mask
			nbresh=tf.reshape(1-noise_partition_utt,[K,1],name='ndresh')
			nbreshV=tf.tile(nbresh,[1,D])
			nbreshV=tf.to_float(nbreshV) # dim: (K x D)
			nbreshY=tf.tile(nbresh,[1,nrS]) # dim: (K x nrS)

			# Reshape the embeddingsvectors
			V=tf.reshape(logits_utt,[K,D],name='V') # dim: (K x D)
			# Normalise Embeddingsvectors
			Vnorm=tf.nn.l2_normalize(V, axis=1, epsilon=1e-12, name='Vnorm')
			# Remove noisy and silent bins
			Vnorm=tf.multiply(tf.multiply(Vnorm,ebreshV),nbreshV) # dim: (K x D)
			# Reshape binary targets and remove noisy and silent bins
			Y=tf.reshape(targets_utt,[K,nrS],name='Y')
			Y=tf.multiply(tf.multiply(Y,ebreshY),nbreshY)
			Y=tf.to_float(Y) # dim: (K x nrS)

			# First part cost function cfr. Hershey et al. 2016
			prod1=tf.matmul(Vnorm,Vnorm,transpose_a=True, transpose_b=False, a_is_sparse=True,
					b_is_sparse=True, name='VTV')
			prod2=tf.matmul(Vnorm,Y,transpose_a=True, transpose_b=False, a_is_sparse=True,
					b_is_sparse=True, name='VTY')

			term1=tf.reduce_sum(tf.square(prod1),name='frob_1')
			term2=tf.reduce_sum(tf.square(prod2),name='frob_2')

			loss_utt_1=tf.add(term1,-2*term2,name='term1and2')
			# Number of non noisy non silence cells squared
			norm_1= tf.square(tf.to_float(tf.reduce_sum(tf.multiply(nbresh,ebresh))))
			norm_1=tf.maximum(norm_1,1)

			# Ideal ratio masker reshape
			ideal_ratio_utt_resh=tf.reshape(ideal_ratio_utt,[K,1],name='ideal_ratio')
			# Alpha output network
			noise_actual=tf.reshape(alpha_utt,[K,1],name='nactual')
			loss_utt_2=tf.reduce_sum(tf.square(ideal_ratio_utt_resh-noise_actual))

			# Normalisation constant: number of cells in spectrogram
			norm_2=tf.to_float(Nspec)
			loss += loss_utt_1/norm_1 + loss_utt_2/norm_2


	norm=tf.to_float(tf.constant(batch_size))
	return loss , norm

def pit_loss(targets, logits, mix_to_mask, seq_length, batch_size,softmax=True,rescale_recs=False):
	"""
	Compute the permutation invariant loss.
	Remark: This is implementation is different from pit_loss as the last dimension of logits is
	still feat_dim*nrS, but the first feat_dim entries correspond to the first speaker and the
	second feat_dim entries correspond to the second speaker and so on. In pit_loss, the first nrS
	entries corresponded to the first feature dimension, the second nrS entries to the seocnd
	feature dimension and so on.
	Remark2: There is actually a more efficient approach to calculate this loss. First calculate
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
	"""

	with tf.name_scope('PIT_loss'):
		feat_dim=targets.get_shape()[2]
		output_dim=logits.get_shape()[2]
		nrS=targets.get_shape()[3]
		permutations=list(itertools.permutations(range(nrS),nrS))

		logits_resh=tf.transpose(tf.reshape(tf.transpose(logits,[2,0,1]),[nrS,feat_dim,batch_size,-1]),[2,3,1,0])
		if softmax:
			Masks=tf.nn.softmax(logits_resh, axis=3)
		else:
			Masks=tf.nn.sigmoid(logits_resh)

		mix_to_mask=tf.expand_dims(mix_to_mask,-1)
		recs=tf.multiply(Masks, mix_to_mask)

		if rescale_recs:
			recs=recs / tf.reduce_max(tf.abs(recs),axis=[1,2],keepdims=True) * \
			  tf.reduce_max(tf.abs(targets),axis=[1,2],keepdims=True)

		targets_resh=tf.transpose(targets,perm=[3,0,1,2])
		recs=tf.transpose(recs,perm=[3,0,1,2])

		norm=tf.to_float(tf.reduce_sum(seq_length)*nrS * feat_dim )

		perm_cost=[]
		for perm in permutations:
			tmp=tf.square(tf.norm(tf.gather(recs,perm)-targets_resh,ord='fro',axis=[2,3]))
			perm_cost.append(tf.reduce_sum(tmp,0))

		loss_utt=tf.reduce_min(perm_cost,0)
		loss=tf.reduce_sum(loss_utt)

	return loss, norm


def cross_entropy_loss_eos(targets, logits, logit_seq_length, target_seq_length):
	"""
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
	"""

	batch_size=tf.shape(targets)[0]

	with tf.name_scope('cross_entropy_loss'):

		output_dim=tf.shape(logits)[2]

		# get the logits for the final timestep
		indices=tf.stack([tf.range(batch_size),
							logit_seq_length-1],
						   axis=1)
		final_logits=tf.gather_nd(logits, indices)

		# stack all the logits except the final logits
		stacked_logits=seq2nonseq(logits,
									logit_seq_length - 1)

		# create the stacked targets
		stacked_targets=seq2nonseq(targets,
									 target_seq_length)

		# create the targets for the end of sequence labels
		final_targets=tf.tile([output_dim-1], [batch_size])

		# add the final logits and targets
		stacked_logits=tf.concat([stacked_logits, final_logits], 0)
		stacked_targets=tf.concat([stacked_targets, final_targets], 0)

		# compute the cross-entropy loss
		losses=tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=stacked_logits,
			labels=stacked_targets)

		loss=tf.reduce_mean(losses)

	return loss


def get_indices(sequence_length):
	"""get the indices corresponding to sequences (and not padding)

	Args:
		sequence_length: the sequence_lengths as a N-D tensor

	Returns:
		A [sum(sequence_length) x N-1] Tensor containing the indices"""

	with tf.name_scope('get_indices'):

		numdims=len(sequence_length.shape)

		# get th emaximal length
		max_length=tf.reduce_max(sequence_length)

		sizes=tf.shape(sequence_length)

		range_tensor=tf.range(max_length-1)
		for i in range(1, numdims):
			tile_dims=[1]*i + [sizes[i]]
			range_tensor=tf.tile(tf.expand_dims(range_tensor, i), tile_dims)

		indices=tf.where(tf.less(range_tensor, tf.expand_dims(sequence_length, numdims)))

	return indices
