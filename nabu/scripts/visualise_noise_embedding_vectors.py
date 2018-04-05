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
from mpl_toolkits.mplot3d import Axes3D

outputdir = '/esat/spchtemp/scratch/r0450064/expDir/gpc/DAN_F_FPC/test/reconstructions/taskF/raw_output'
centerdir = '/esat/spchtemp/scratch/r0450064/expDir/gpc/DAN_F_FPC/test/reconstructions/taskF/cluster_centers'
expdir = '/esat/spchtemp/scratch/r0450064/expDir/gpc/DAN_F_FPC/test'
#expdir_noise = '/esat/spchtemp/scratch/r0450064/expDir/wsj/DC_WSJ_NOISE/test'
task = 'taskF'
nrS = 2

evaluator_cfg = configparser.ConfigParser()
evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

#noise_evaluator_cfg = configparser.ConfigParser()
#noise_evaluator_cfg.read(os.path.join(expdir_noise,'evaluator.cfg'))

task_evaluator_cfg = dict(evaluator_cfg.items(task))
#noise_task_evaluator_cfg = dict(noise_evaluator_cfg.items(task))

database_cfg = configparser.ConfigParser()
database_cfg.read(os.path.join(expdir, 'database.cfg'))

#noise_database_cfg = configparser.ConfigParser()
#noise_database_cfg.read(os.path.join(expdir_noise,'database.cfg'))

#noise_targets_name = noise_task_evaluator_cfg['noise_targets']
#noise_targets_dataconf = dict(noise_database_cfg.items(noise_targets_name))

binary_targets_name = task_evaluator_cfg['partition_targets']
binary_targets_dataconf = dict(database_cfg.items(binary_targets_name))

usedbin_targets_name = task_evaluator_cfg['usedbins']
usedbin_targets_dataconf = dict(database_cfg.items(usedbin_targets_name))

#noise_targets_reader = data_reader.DataReader(noise_targets_dataconf)
usedbin_targets_reader = data_reader.DataReader(usedbin_targets_dataconf)
binary_targets_reader = data_reader.DataReader(binary_targets_dataconf)

i = 2719

#noise_targets_complete, utt_info = noise_targets_reader(i)
#noise_targets = noise_targets_complete[:,nrS::nrS+1]
#noise_targets_shape = np.shape(noise_targets)
#noise_targets = np.reshape(noise_targets,noise_targets_shape[0]*noise_targets_shape[1])
usedbins,utt_info= usedbin_targets_reader(i)
usedbins_shape = np.shape(usedbins)
usedbins = np.reshape(usedbins,usedbins_shape[0]*usedbins_shape[1])
emb_vec = np.load(os.path.join(outputdir,'bin_emb_'+utt_info['utt_name']+'.npy'))
center = np.load(os.path.join(centerdir,utt_info['utt_name']+'.npy'))
#if noise_targets_shape[0] != np.shape(emb_vec)[0]:
#    raise Exception('Incorrect dimension')

binary_targets_complete,_ = binary_targets_reader(i)
binary_targets_spk1 = binary_targets_complete[:,::nrS]
binary_targets_spk1 = np.reshape(binary_targets_spk1,usedbins_shape[0]*usedbins_shape[1])
binary_targets_spk2 = binary_targets_complete[:,1::nrS]
binary_targets_spk2 = np.reshape(binary_targets_spk2,usedbins_shape[0]*usedbins_shape[1])


emb_vec = np.reshape(emb_vec,[usedbins_shape[0]*usedbins_shape[1],20])
#emb_vec_norm = np.linalg.norm(emb_vec,axis=1,keepdims=True)
#emb_vec = emb_vec/emb_vec_norm

#emb_vec_spk1 = emb_vec[np.logical_and(np.logical_not(noise_targets),binary_targets_spk1)]
#emb_vec_spk2 = emb_vec[np.logical_and(np.logical_not(noise_targets),binary_targets_spk2)]
pca = PCA(n_components = 2)
#clf = LinearDiscriminantAnalysis(n_components=1)
#clf.fit(np.vstack((emb_vec_spk1,emb_vec_spk2)),np.vstack((np.zeros((np.shape(emb_vec_spk1)[0],1)),np.ones((np.shape(emb_vec_spk2)[0],1)))))

#emb_vec_proj = clf.transform(emb_vec)
emb_vec_proj = pca.fit_transform(emb_vec)
center_proj = pca.transform(center)
#print emb_vec_proj
#emb_vec_proj_noise = emb_vec_proj[noise_targets]
#emb_vec_proj_spk1 = emb_vec_proj[np.logical_and(np.logical_not(noise_targets),binary_targets_spk1)]
#emb_vec_proj_spk2 = emb_vec_proj[np.logical_and(np.logical_not(noise_targets),binary_targets_spk2)]

emb_vec_proj_spk1 = emb_vec_proj[np.logical_and(usedbins,binary_targets_spk1)]
emb_vec_proj_spk2 = emb_vec_proj[np.logical_and(usedbins,binary_targets_spk2)]

fig = plt.figure(1)
#plt.plot(emb_vec_proj_noise[:,0],emb_vec_proj_noise[:,1],'b.')
plt.plot(emb_vec_proj_spk1[:,0],emb_vec_proj_spk1[:,1],'r.',markersize=1)
plt.plot(emb_vec_proj_spk2[:,0],emb_vec_proj_spk2[:,1],'g.',markersize=1)
plt.plot(center_proj[:,0],center_proj[:,1],'kx')
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(emb_vec_proj_noise[:,0],emb_vec_proj_noise[:,1],emb_vec_proj_noise[:,2],'b.')
#plt.hold(True)
#ax.scatter(emb_vec_proj_spk1[:,0],emb_vec_proj_spk1[:,1],emb_vec_proj_spk1[:,2],'r.',s=1)
#ax.scatter(emb_vec_proj_spk2[:,0],emb_vec_proj_spk2[:,1],emb_vec_proj_spk2[:,2],'g.',s=1)
#plt.savefig('foo.png')
plt.show()
