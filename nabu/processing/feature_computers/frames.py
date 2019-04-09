'''@file frames.py
contains the frames computer'''

import numpy as np
import base
import feature_computer
from sigproc import snip

class Frames(feature_computer.FeatureComputer):
    '''the feature computer class to compute frames'''

    def comp_feat(self, sig, rate):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

        #snip the edges
        sig = snip(sig, rate, float(self.conf['winlen']),
                   float(self.conf['winstep']))

        feat = base.frames(sig, rate, self.conf)


        return feat

    def get_dim(self):
        '''the feature dimemsion'''

        dim = int(self.conf['l'])

        return dim
