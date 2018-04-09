'''@file sdr_scorer.py
contains the scorer using SDR'''

import scorer
import numpy as np
import os
import scipy.io.wavfile as wav
from nabu.postprocessing import data_reader
import time
import bss_eval
import pdb

class SdrSnrScorer(object):
    '''the SDR scorer class. Uses the script from
    C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis,
    'mir_eval: A Transparent Implementation of Common MIR Metrics', Proceedings of the 15th
    International Conference on Music Information Retrieval, 2014

    a scorer using SDR'''

    score_metrics = ('SDR','SIR','SNR','SAR','perm')
    score_metrics_to_summarize = ('SDR','SIR','SNR','SAR')
    score_scenarios = ('SS','base')
    score_expects = 'data'

    def __init__(self, conf, evalconf, dataconf, rec_dir, numbatches, task):
        '''Reconstructor constructor
        Args:
            conf: the scorer configuration as a dictionary
            evalconf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            rec_dir: the directory where the reconstructions are
            numbatches: the number of batches to process
        '''

        if evalconf.has_option(task,'batch_size'):
            batch_size = int(evalconf.get(task,'batch_size'))
        else:
            batch_size = int(evalconf.get('evaluator','batch_size'))
        self.tot_utt = batch_size * numbatches
        print batch_size,self.tot_utt

        self.rec_dir = rec_dir
        self.segment_lengths = evalconf.get('evaluator','segment_length').split(' ')

        #get the original source signals reader
        org_src_name = conf['org_src']
        org_src_dataconf = dict(dataconf.items(org_src_name))
        self.org_src_reader = data_reader.DataReader(org_src_dataconf,self.segment_lengths)

        #get the base signal (original mixture) reader
        base_name = conf['base']
        base_dataconf = dict(dataconf.items(base_name))
        self.base_reader = data_reader.DataReader(base_dataconf,self.segment_lengths)

        #get the speaker info
        spkinfo_name = conf['spkinfo']
        spkinfo_dataconf = dict(dataconf.items(spkinfo_name))
        spkinfo_file = spkinfo_dataconf['datafiles']

        self.utt_spkinfo = dict()
        for line in open(spkinfo_file):
            splitline = line.strip().split(' ')
            utt_name = splitline[0]
            dataline = ' '.join(splitline[2:])
            self.utt_spkinfo[utt_name] = dataline

        #predefined mixture types
        self.mix_types = ['all_m', 'all_f','same_gen', 'diff_gen']

        #create the dictionary where all results will be stored
        self.results = dict()

        #metrics to be used in sumarize function, if not yet stated
        if not self.score_metrics_to_summarize:
               self.score_metrics_to_summarize = self.score_metrics

        noise_name = conf['noise']
        noise_dataconf = dict(dataconf.items(noise_name))
        self.noise_reader = data_reader.DataReader(noise_dataconf,self.segment_lengths)
    def __call__(self):
        ''' score the utterances in the reconstruction dir with the original source signals
        '''

        for utt_ind in range(self.tot_utt):
            if np.mod(utt_ind,10) == 0:
                print 'Getting results for utterance %d' %utt_ind

            if self.score_expects == 'data':
            #Gather the data for scoring

                #get the source signals
                org_src_signals, utt_info = self.org_src_reader(utt_ind)

                nrS = utt_info['nrSig']
                utt_name = utt_info['utt_name']

                #get the base signal (original mixture) and duplicate it
                base_signal, _ = self.base_reader(utt_ind)

                base_signals = list()
                for spk in range(nrS):
                    base_signals.append(base_signal)
                noise_signal, _ = self.noise_reader(utt_ind)
                #get the reconstructed signals
                rec_src_signals = list()
                rec_src_filenames = list()
                for spk in range(nrS):
                    filename = os.path.join(self.rec_dir,'s'+str(spk+1),utt_name+'.wav')
                    _, utterance = wav.read(filename)
                    rec_src_signals.append(utterance)
                    rec_src_filenames.append(filename)

                #get the scores for the utterance (in dictionary format)
                utt_score_dict = self._get_score(org_src_signals, base_signals,
                           rec_src_signals,noise_signal)

            elif self.score_expects == 'files':
                #Gather the filnames for scoring

                splitline = self.org_src_reader.datafile_lines[utt_ind].strip().split(' ')
                utt_name = splitline[0]
                org_src_filenames = splitline[1:]
                nrS = len(org_src_filenames)

                splitline = self.base_reader.datafile_lines[utt_ind].strip().split(' ')
                base_filename = splitline[1]
                base_filenames = list()
                for spk in range(nrS):
                    base_filenames.append(base_filename)

                splitline = self.noise_reader.datafile_lines[utt_ind].strip().split(' ')
                noise_filename = splitline[1]
                rec_src_filenames = list()
                for spk in range(nrS):
                    filename = os.path.join(self.rec_dir,'s'+str(spk+1),utt_name+'.wav')
                    rec_src_filenames.append(filename)

                #get the scores for the utterance (in dictionary format)
                utt_score_dict = self._get_score(org_src_filenames, base_filenames,
                           rec_src_filenames,noise_filename)

            else:
                raise Exception('unexpected input for scrorer_expects: %s' %self.score_expects)



            #get the speaker info
            spk_info = dict()
            spk_info['ids']=[]
            spk_info['genders']=[]
            dataline = self.utt_spkinfo[utt_name]
            splitline = dataline.strip().split(' ')
            for spk in range(nrS):
                spk_info['ids'].append(splitline[spk*2])
                spk_info['genders'].append(splitline[spk*2+1])

            spk_info['mix_type']=dict()
            for mix_type in self.mix_types:
                spk_info['mix_type'][mix_type]=False
            if all(gender=='M' for gender in spk_info['genders']):
                spk_info['mix_type']['all_m']=True
                spk_info['mix_type']['same_gen']=True
            elif all(gender=='F' for gender in spk_info['genders']):
                spk_info['mix_type']['all_f']=True
                spk_info['mix_type']['same_gen']=True
            else:
                spk_info['mix_type']['diff_gen']=True

            #assemble results
            self.results[utt_name]=dict()
            self.results[utt_name]['score']=utt_score_dict
            self.results[utt_name]['spk_info']=spk_info


    def summarize(self):
        '''summarize the results of all utterances
    '''

    #

        utts = self.results.keys()

        mix_type_indeces = dict()
        for mix_type in self.mix_types:
            mix_type_indeces[mix_type]=[]

            for i,utt in enumerate(utts):
                if self.results[utt]['spk_info']['mix_type'][mix_type]:
                    mix_type_indeces[mix_type].append(i)

        result_summary = dict()
        for metric in self.score_metrics_to_summarize:
            result_summary[metric] = dict()

            for scen in self.score_scenarios:
                result_summary[metric][scen] = dict()

                tmp = []
                for i,utt in enumerate(utts):

                    utt_score = np.mean(self.results[utt]['score'][metric][scen])
                    tmp.append(utt_score)

                result_summary[metric][scen]['all'] = np.mean(tmp)

                for mix_type in self.mix_types:
                    inds = mix_type_indeces[mix_type]
                    result_summary[metric][scen][mix_type] = np.mean([tmp[i] for i in inds])
            #
        for metric in self.score_metrics_to_summarize:
            print ''
            print 'Result for %s (using %s): ' % (metric,self.__class__.__name__)

            for mix_type in ['all']+self.mix_types:
                print 'for %s: ' % mix_type,

                for scen in self.score_scenarios:
                    print '%f (%s),' % (result_summary[metric][scen][mix_type],scen),
                #if only 2 scenarios, print the difference
                if len(self.score_scenarios)==2:
                    scen1 = self.score_scenarios[0]
                    scen2 = self.score_scenarios[1]
                    diff = result_summary[metric][scen1][mix_type] - result_summary[metric][scen2][mix_type]
                    print '%f (absolute difference)' %diff
                else:
                    print ''

        return result_summary
    def _get_score(self,org_src_signals, base_signals, rec_src_signals, noise_signal):
        '''score the reconstructed utterances with respect to the original source signals

        Args:
            org_src_signals: the original source signals, as a list of numpy arrarys
            base_signals: the duplicated base signal (original mixture), as a list of numpy arrarys
            rec_src_signals: the reconstructed source signals, as a list of numpy arrarys

        Returns:
            the score'''

        #convert to numpy arrays
        org_src_signals = np.array(org_src_signals)[:,:,0]
        base_signals=np.array(base_signals)[:,:,0]
        rec_src_signals=np.array(rec_src_signals)
        noise_signal = np.squeeze(noise_signal)
        #
        collect_outputs=dict()
        collect_outputs[self.score_scenarios[1]] = bss_eval.bss_eval_sources_extended(org_src_signals,base_signals,noise_signal)
        collect_outputs[self.score_scenarios[0]] = bss_eval.bss_eval_sources_extended(org_src_signals,rec_src_signals,noise_signal)

        nrS=len(org_src_signals)

        #convert the outputs to a single dictionary
        score_dict = dict()
        for i,metric in enumerate(self.score_metrics):
            score_dict[metric]=dict()

            for j,scen in enumerate(self.score_scenarios):
                score_dict[metric][scen]=[]

                for spk in range(nrS):
                    score_dict[metric][scen].append(collect_outputs[scen][i][spk])

        return score_dict
