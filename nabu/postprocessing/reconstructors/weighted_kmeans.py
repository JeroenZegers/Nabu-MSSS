# Based on: https://towardsdatascience.com/clustering-the-us-population-observation-weighted-k-means-f4d58b370002

import random
import numpy as np
import scipy.spatial


def distance(p1,p2):
  return np.linalg.norm(p1,p2)

def cluster_centroids(data,weights, clusters, k):
  results=[]
  for i in range(k):
    results.append( np.average(data[clusters == i],weights=weights[clusters == i],axis=0))
  return np.array(results)

def kmeans(data,weights, k, steps=20):
  if(np.shape(data)[0] != np.shape(weights)[0]):
      print "Dimension data and weights don't match"
  # Forgy initialization method: choose k data points randomly.
  centroids = data[np.random.choice(np.arange(len(data)), k, False)]

  for _ in range(max(steps, 1)):
    sqdists = scipy.spatial.distance.cdist(centroids, data, 'euclidean')

    # Index of the closest centroid to each data point.
    clusters = np.argmin(sqdists, axis=0)

    new_centroids = cluster_centroids(data,weights, clusters, k)

    if np.array_equal(new_centroids, centroids):
      break

    centroids = new_centroids

  return clusters, centroids
