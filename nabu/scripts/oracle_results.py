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

def reconstruct_signals(mixture,noise_mask,bin_targets_1,bin_targets_2,utt_info,org_mix_reader):
    '''reconstruct the signals

        Args:
            inputs: the output of a single utterance of the neural network

        Returns:
            the reconstructed signals
            some info on the utterance'''
    denoised_mixture = mixture*noise_mask        
    reconstructed_signals = list()
    sig_1 = denoised_mixture*bin_targets_1
    reconstructed_signals.append(base.spec2time(sig_1, utt_info['rate'], 
                           utt_info['siglen'],
                           org_mix_reader.processor.comp.conf))
        
    sig_2 = denoised_mixture*bin_targets_2
    reconstructed_signals.append(base.spec2time(sig_2, utt_info['rate'], 
                           utt_info['siglen'],
                           org_mix_reader.processor.comp.conf))
    return reconstructed_signals

expdir = '/esat/spchtemp/scratch/r0450064/expDir/wsj/DC_WSJ_NOISE/test'
task = 'task0'
rec_dir ='/esat/spchtemp/scratch/r0450064/expDir/wsj/DC_WSJ_NOISE/test/oracle'
nrS = 2

evaluator_cfg = configparser.ConfigParser()
evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

reconstructor_cfg = configparser.ConfigParser()
reconstructor_cfg.read(os.path.join(expdir,'reconstructor.cfg'))

task_evaluator_cfg = dict(evaluator_cfg.items(task))
task_reconstructor_cfg = dict(reconstructor_cfg.items(task))

database_cfg = configparser.ConfigParser()
database_cfg.read(os.path.join(expdir, 'database.cfg'))

noise_targets_name = task_evaluator_cfg['noise_targets']
noise_targets_dataconf = dict(database_cfg.items(noise_targets_name))

binary_targets_name = task_evaluator_cfg['binary_targets']
binary_targets_dataconf = dict(database_cfg.items(binary_targets_name))

org_mix_name = task_reconstructor_cfg['org_mix']
org_mix_dataconf = dict(database_cfg.items(org_mix_name))

noise_targets_reader = data_reader.DataReader(noise_targets_dataconf)
binary_targets_reader = data_reader.DataReader(binary_targets_dataconf)
org_mix_reader = data_reader.DataReader(org_mix_dataconf)


#if not os.path.isdir(os.path.join(rec_dir)):
#    os.makedirs(os.path.join(rec_dir))
#scp_file = open(os.path.join(rec_dir,'pointers.scp'),'w+')

# Make storage folders
#for spk in range(nrS):
#    if not os.path.isdir(os.path.join(rec_dir,'s' + str(spk+1))):
#        os.makedirs(os.path.join(rec_dir,'s' + str(spk+1)))

#for i in range(0,3000):
#    if i%10 == 0:
#        print "reconstructing ",i
#    noise_targets_complete,_ = noise_targets_reader(i)
#    noise_targets = noise_targets_complete[:,nrS::nrS+1]
#    noise_targets_shape = np.shape(noise_targets)
#    binary_targets,_ = binary_targets_reader(i)
#    binary_targets_spk1 = binary_targets[:,::nrS].astype(int)
#    binary_targets_spk2 = binary_targets[:,1::nrS].astype(int)
#    binary_targets_shape = np.shape(binary_targets)
#    mixture,utt_info = org_mix_reader(i)
#    mixture_shape = np.shape(mixture)

#    if binary_targets_shape[0] != noise_targets_shape[0] or binary_targets_shape[1] != noise_targets_shape[1]*2:
#        raise Exception("Dimension error, binary targets")
#    if mixture_shape[0] != noise_targets_shape[0] or mixture_shape[1] != noise_targets_shape[1]:
#        raise Exception("Dimension error, original mixture")
#    noise_mask = 1 - noise_targets.astype(int)

#    rs = reconstruct_signals(mixture,noise_mask,binary_targets_spk1,binary_targets_spk2,utt_info,org_mix_reader)
#   write_audiofile(rec_dir,scp_file,nrS,rs, utt_info) 

scorer_cfg = configparser.ConfigParser()
scorer_cfg.read(os.path.join(expdir,'scorer.cfg'))
task_scorer_cfg = dict(scorer_cfg.items(task))
scorer = SdrScorer(task_scorer_cfg,evaluator_cfg,database_cfg,rec_dir,100,task)
scorer()
scorer.summarize()

