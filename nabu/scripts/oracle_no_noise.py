# Calculate oracle performance for noise free mixtures

import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from six.moves import configparser
from nabu.postprocessing import data_reader
from nabu.postprocessing.scorers.sdr_scorer import SdrScorer
from nabu.processing.feature_computers import base
import scipy.io.wavfile as wav


def write_audiofile(rec_dir,scp_file,nrS, reconstructed_signals, utt_info):
    '''write the audiofiles for the reconstructions

        Args:
            reconstructed_signals: the reconstructed signals for a single mixture
            utt_info: some info on the utterance
    '''

    write_str=utt_info['utt_name']
    for spk in range(nrS):
        rec_dir_spk = os.path.join(rec_dir,'s' + str(spk+1))
        filename = os.path.join(rec_dir_spk,utt_info['utt_name']+'.wav')

        signal = reconstructed_signals[spk]
        wav.write(filename, utt_info['rate'], signal)
        write_str += ' ' + filename

    write_str += ' \n'
    scp_file.write(write_str)

def reconstruct_signals(mixture,bin_targets_1,bin_targets_2,utt_info,org_mix_reader):
    '''reconstruct the signals

        Args:
            inputs: the output of a single utterance of the neural network

        Returns:
            the reconstructed signals
            some info on the utterance'''
    reconstructed_signals = list()
    sig_1 = mixture*bin_targets_1
    reconstructed_signals.append(base.spec2time(sig_1, utt_info['rate'],
                           utt_info['siglen'],
                           org_mix_reader.processor.comp.conf))

    sig_2 = mixture*bin_targets_2
    reconstructed_signals.append(base.spec2time(sig_2, utt_info['rate'],
                           utt_info['siglen'],
                           org_mix_reader.processor.comp.conf))
    return reconstructed_signals

expdir = '/esat/spchtemp/scratch/r0450064/expDir/gpc/Orakel_Z_ZAS/test'
task = 'taskS'
rec_dir ='/esat/spchtemp/scratch/r0450064/expDir/gpc/Orakel_Z_ZAS/test/oracle/taskS'
nrS = 2

## Test evaluator
evaluator_cfg = configparser.ConfigParser()
evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

# Reconstructor
reconstructor_cfg = configparser.ConfigParser()
reconstructor_cfg.read(os.path.join(expdir,'reconstructor.cfg'))

# Section of task
task_evaluator_cfg = dict(evaluator_cfg.items(task))
task_reconstructor_cfg = dict(reconstructor_cfg.items(task))

# General database file
database_cfg = configparser.ConfigParser()
database_cfg.read(os.path.join(expdir, 'database.cfg'))

# partition targets (ground truth)
partition_targets_name = task_evaluator_cfg['binary_targets']
partition_targets_dataconf = dict(database_cfg.items(partition_targets_name))

# spectrogram original mixture
org_mix_name = task_reconstructor_cfg['org_mix']
org_mix_dataconf = dict(database_cfg.items(org_mix_name))

# read from dataforTF files
partition_targets_reader = data_reader.DataReader(partition_targets_dataconf)
org_mix_reader = data_reader.DataReader(org_mix_dataconf)

# Create reconstruction directory
if not os.path.isdir(os.path.join(rec_dir)):
    os.makedirs(os.path.join(rec_dir))
# File with all files
scp_file = open(os.path.join(rec_dir,'pointers.scp'),'w+')

# Make storage folders
for spk in range(nrS):
    if not os.path.isdir(os.path.join(rec_dir,'s' + str(spk+1))):
        os.makedirs(os.path.join(rec_dir,'s' + str(spk+1)))
for i in range(0,3000):
    if i%10 == 0:
        print "reconstructing ",i
    partition_targets,_ = partition_targets_reader(i)
    partition_targets_spk1 = partition_targets[:,::nrS].astype(int)
    partition_targets_spk2 = partition_targets[:,1::nrS].astype(int)
    assert np.shape(partition_targets_spk1) == np.shape(partition_targets_spk2)
    mixture,utt_info = org_mix_reader(i)
    mixture_shape = np.shape(mixture)
    assert np.shape(partition_targets_spk1) == mixture_shape
    rs = reconstruct_signals(mixture,partition_targets_spk1,partition_targets_spk2,utt_info,org_mix_reader)
    write_audiofile(rec_dir,scp_file,nrS,rs, utt_info)

scorer_cfg = configparser.ConfigParser()
scorer_cfg.read(os.path.join(expdir,'scorer.cfg'))
task_scorer_cfg = dict(scorer_cfg.items(task))
scorer = SdrScorer(task_scorer_cfg,evaluator_cfg,database_cfg,rec_dir,100,task)
scorer()
scorer.summarize()
