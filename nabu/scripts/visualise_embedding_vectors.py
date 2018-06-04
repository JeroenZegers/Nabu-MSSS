# Script to visualise the embeddings vectors
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from six.moves import configparser
from nabu.postprocessing import data_reader
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

rawoutputdir = '/esat/spchtemp/scratch/r0450064/expDir/gpc/DAN_F_FPC/test/reconstructions/taskF/raw_output'
centerdir = '/esat/spchtemp/scratch/r0450064/expDir/gpc/DAN_F_FPC/test/reconstructions/taskF/cluster_centers'
expdir = '/esat/spchtemp/scratch/r0450064/expDir/gpc/DAN_F_FPC/test'
task = 'taskF'
nrS = 2

evaluator_cfg = configparser.ConfigParser()
evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

task_evaluator_cfg = dict(evaluator_cfg.items(task))

database_cfg = configparser.ConfigParser()
database_cfg.read(os.path.join(expdir, 'database.cfg'))

binary_targets_name = task_evaluator_cfg['partitioning']
binary_targets_dataconf = dict(database_cfg.items(binary_targets_name))

energybin_targets_name = task_evaluator_cfg['energybins']
energybin_targets_dataconf = dict(database_cfg.items(energybin_targets_name))

energybin_targets_reader = data_reader.DataReader(energybin_targets_dataconf)
binary_targets_reader = data_reader.DataReader(binary_targets_dataconf)

i = 2719


energybins,utt_info= energybin_targets_reader(i)
energybins_shape = np.shape(energybins)
energybins = np.reshape(energybins,energybins_shape[0]*energybins_shape[1])
emb_vec = np.load(os.path.join(rawoutputdir,'emb_vec_'+utt_info['utt_name']+'.npy'))
center = np.load(os.path.join(centerdir,utt_info['utt_name']+'.npy'))


binary_targets_complete,_ = binary_targets_reader(i)
binary_targets_spk1 = binary_targets_complete[:,::nrS]
binary_targets_spk1 = np.reshape(binary_targets_spk1,energybins_shape[0]*energybins_shape[1])
binary_targets_spk2 = binary_targets_complete[:,1::nrS]
binary_targets_spk2 = np.reshape(binary_targets_spk2,energybins_shape[0]*energybins_shape[1])


emb_vec = np.reshape(emb_vec,[energybins_shape[0]*energybins_shape[1],20])

pca = PCA(n_components = 2)

emb_vec_proj = pca.fit_transform(emb_vec)
center_proj = pca.transform(center)

emb_vec_proj_spk1 = emb_vec_proj[np.logical_and(energybins,binary_targets_spk1)]
emb_vec_proj_spk2 = emb_vec_proj[np.logical_and(energybins,binary_targets_spk2)]

fig = plt.figure(1)

plt.plot(emb_vec_proj_spk1[:,0],emb_vec_proj_spk1[:,1],'r.',markersize=1)
plt.plot(emb_vec_proj_spk2[:,0],emb_vec_proj_spk2[:,1],'g.',markersize=1)
plt.plot(center_proj[:,0],center_proj[:,1],'kx')
plt.show()
