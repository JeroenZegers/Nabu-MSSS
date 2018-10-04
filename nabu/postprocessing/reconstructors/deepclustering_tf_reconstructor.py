'''@file deepclustering_tf_reconstructor.py
contains the reconstor class using deep clustering in tensorflow implementation'''

#from sklearn.cluster import KMeans
import tensorflow as tf
import mask_reconstructor
from nabu.postprocessing import data_reader
import numpy as np
import os
import clustering_ops_jeroen
import pdb

class DeepclusteringTFReconstructor(mask_reconstructor.MaskReconstructor):
    '''the deepclustering reconstructor class

    a reconstructor using deep clustering with tensorflow implementation'''
    
    requested_output_names = ['bin_emb']

    def __init__(self, conf, evalconf, dataconf, rec_dir, task):
        '''DeepclusteringTFReconstructor constructor

        Args:
            conf: the reconstructor configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions will be stored
        '''
        
        super(DeepclusteringTFReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task)
        
        #get the usedbins reader
        usedbins_name = conf['usedbins']
        usedbins_dataconf = dict(dataconf.items(usedbins_name))
        self.usedbins_reader = data_reader.DataReader(usedbins_dataconf,self.segment_lengths)
        
        #directory where cluster centroids will be stored
        self.center_store_dir = os.path.join(rec_dir,'cluster_centers')
        if not os.path.isdir(self.center_store_dir):
	      os.makedirs(self.center_store_dir)
        

    def _get_masks(self, output, utt_info):
	'''estimate the masks

	Args:
	    output: the output of a single utterance of the neural network
            utt_info: some info on the utterance

	Returns:
	    the estimated masks'''
 
	embeddings = output['bin_emb']
	#only the non-silence bins will be used for the clustering    
	usedbins, _ = self.usedbins_reader(self.pos)
	
	[T,F] = np.shape(usedbins)
	emb_dim = np.shape(embeddings)[1]/F
	
	if np.shape(embeddings)[0] != T:
	    raise 'Number of frames in usedbins does not match the sequence length'
	
	#reshape the outputs
	embeddings = embeddings[:T,:]
	embeddings_resh = np.reshape(embeddings,[T*F,emb_dim])
	embeddings_resh_norm = np.linalg.norm(embeddings_resh,axis=1,keepdims=True)
	embeddings_resh = embeddings_resh/embeddings_resh_norm
	
	#only keep the active bins (above threshold) for clustering
	usedbins_resh = np.reshape(usedbins, T*F)
	embeddings_speech_resh = embeddings_resh[usedbins_resh]
	
	best_kmeans_score=1e20
	best_kmeans_cluster_centers_var_value=None
	for repeat_ind in range(10):
	    graph_kmeans = tf.Graph()
	    with graph_kmeans.as_default():
		in_placeholder = tf.placeholder(tf.float32, shape=np.shape(embeddings_speech_resh))
		random_seed=int(np.random.rand(1)*1000)
		kmeans = clustering_ops_jeroen.KMeans(
		    inputs=in_placeholder,
		    num_clusters=self.nrS,
		    #initial_clusters="random",
		    initial_clusters="kmeans_plus_plus",
		    distance_metric='squared_euclidean',
		    random_seed=random_seed,
		    use_mini_batch=True)
		
		(all_scores, cluster_idx, scores, cluster_centers_initialized,
		    cluster_centers_var, init_op, training_op) = kmeans.training_graph()
		cluster_idx=cluster_idx[0]

		init = tf.global_variables_initializer()
		loss=tf.reduce_mean(scores)
		
		with tf.Session() as sess:
		    sess.run(init)
		    feed_dict={in_placeholder:embeddings_speech_resh}
		    sess.run(init_op,feed_dict=feed_dict)
		    cluster_idx_val_old=None
		    for i in range(1000):
			fetch=[training_op,loss,cluster_centers_var,cluster_idx]
			_,loss_value,cluster_centers_var_value,cluster_idx_val=sess.run(fetch,
					    feed_dict=feed_dict)

			if i>100 and all(cluster_idx_val==cluster_idx_val_old):
			    break
			cluster_idx_val_old=cluster_idx_val
		    
	    if loss_value < best_kmeans_score:
		best_kmeans_score = loss_value
		best_kmeans_cluster_centers_var_value=cluster_centers_var_value
	
	#make a new kmeans where the clusters are initialized with 
	graph_eval_best_kmeans = tf.Graph()
	with graph_eval_best_kmeans.as_default():
	    #full_in_placeholder = tf.placeholder(tf.float32, shape=np.shape(embeddings_resh))
	    kmeans = clustering_ops_jeroen.KMeans(
		  inputs=tf.constant(embeddings_resh),
		  num_clusters=self.nrS,
		  initial_clusters=tf.constant(best_kmeans_cluster_centers_var_value),
		  #initial_clusters='random',
		  distance_metric='squared_euclidean',
		  use_mini_batch=True)
	    (_, cluster_idx, scores, _, _, init_op, _) = kmeans.training_graph()
	    cluster_idx=cluster_idx[0]
	    init = tf.global_variables_initializer()
	    loss=tf.reduce_mean(scores)

	    with tf.Session() as sess_full:
		sess_full.run(init)
		#feed_dict={full_in_placeholder:embeddings_resh}
		sess_full.run(init_op)#,feed_dict=feed_dict)
		
		predicted_labels,loss_value=\
		  sess_full.run([cluster_idx,loss])#,feed_dict=feed_dict)
	

	predicted_labels_resh = np.reshape(predicted_labels,[T,F])
	
	#reconstruct the masks from the cluster labels
	masks = np.zeros([self.nrS,T,F])
	for spk in range(self.nrS):
	    masks[spk,:,:] = predicted_labels_resh==spk
	    
	#store the clusters
	np.save(os.path.join(self.center_store_dir,utt_info['utt_name']),cluster_centers_var_value)
	
	
	
	return masks