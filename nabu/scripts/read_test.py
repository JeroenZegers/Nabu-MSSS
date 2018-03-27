import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from six.moves import configparser
from nabu.postprocessing import data_reader
from sklearn.metrics import roc_curve,auc

outputdir = '/esat/spchtemp/scratch/r0450064/expDir/wsj/DC_WSJ_NOISE2/test/reconstructions/task0/raw_output'
expdir = '/esat/spchtemp/scratch/r0450064/expDir/wsj/DC_WSJ_NOISE2/test'
task = 'task0'
nrS = 2

thresholds = [0.25,0.4,0.5,0.6,0.7]
miss_class = [0,0,0,0,0]
miss_class_usedbins = [0,0,0,0,0]
length = 0
length_used = 0
evaluator_cfg = configparser.ConfigParser()
evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))


task_evaluator_cfg = dict(evaluator_cfg.items(task))

database_cfg = configparser.ConfigParser()
database_cfg.read(os.path.join(expdir, 'database.cfg'))

noise_targets_name = task_evaluator_cfg['noise_targets']
noise_targets_dataconf = dict(database_cfg.items(noise_targets_name))

usedbin_targets_name = task_evaluator_cfg['usedbins']
usedbin_targets_dataconf = dict(database_cfg.items(usedbin_targets_name))

noise_targets_reader = data_reader.DataReader(noise_targets_dataconf)
usedbin_targets_reader = data_reader.DataReader(usedbin_targets_dataconf)

for i in range(0,3000):
    noise_targets_complete, utt_info= noise_targets_reader(i)
    noise_targets = noise_targets_complete[:,nrS::nrS+1]
    noise_targets_shape = np.shape(noise_targets)
    noise_targets = np.reshape(noise_targets.astype(int),noise_targets_shape[0]*noise_targets_shape[1])
    usedbins,_= usedbin_targets_reader(i)
    usedbins = np.reshape(usedbins,noise_targets_shape[0]*noise_targets_shape[1])
    noise_labels = np.load(os.path.join(outputdir,'noise_labels_'+utt_info['utt_name']+'.npy'))
    noise_labels_shape = np.shape(noise_labels)
    if noise_labels_shape[0] != noise_targets_shape[0] or noise_labels_shape[1] != noise_targets_shape[1]:
        print "Dimension error"
        break;
    noise_labels =np.reshape(noise_labels,noise_targets_shape[0]*noise_targets_shape[1])
    for j in range(len(thresholds)):
        miss_class[j] += np.sum(np.abs(noise_targets-(noise_labels>thresholds[j]).astype(int)))
        miss_class_usedbins[j] += np.sum(np.abs(noise_targets-(noise_labels>thresholds[j]).astype(int))[usedbins])
    length += np.shape(noise_targets)[0]
    length_used += np.shape(noise_targets[usedbins])[0]
    

for j in range(len(thresholds)):
    print thresholds[j],float(miss_class[j])/length
    print thresholds[j],float(miss_class_usedbins[j])/length_used
