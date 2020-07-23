"""@file spec.py
contains the complex spectrum computer"""

import numpy as np
import base
import feature_computer
from sigproc import snip


class Spec(feature_computer.FeatureComputer):
    """the computer class to compute complex spectrum"""

    def comp_feat(self, sig, rate):
        """
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        """

        # snip the edges
        sig = snip(sig, rate, float(self.conf['winlen']),
                   float(self.conf['winstep']))

        if 'scipy' in self.conf and self.conf['scipy'] == 'True':
            feat = base.spec_scipy(sig, rate, self.conf)
        else:
            feat = base.spec(sig, rate, self.conf)

        return feat

    def get_dim(self):
        """the feature dimemsion"""

        dim = int(self.conf['nfft'])/2+1

        return dim
