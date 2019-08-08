"""@file angspec.py
contains the angular spectrum feature computer"""

import numpy as np
import base
import feature_computer
from sigproc import snip


class Angspec(feature_computer.FeatureComputer):
    """the feature computer class to compute angular spectrum feature"""

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
        sig = snip(sig, rate, float(self.conf['winlen']), float(self.conf['winstep']))

        feat = base.angspec(sig, rate, self.conf)

        if self.conf['include_energy'] == 'True':
            _, energy = base.fbank(sig, rate, self.conf)
            feat = np.append(feat, energy[:, np.newaxis], 1)

        return feat

    def get_dim(self):
        """the feature dimemsion"""

        dim = int(self.conf['nfft'])/2+1

        if self.conf['include_energy'] == 'True':
            dim += 1

        return dim
