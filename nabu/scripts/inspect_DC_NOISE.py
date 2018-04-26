import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from six.moves import configparser
from nabu.postprocessing import data_reader
from nabu.processing.feature_computers import base




expdir = 'config/recipes/SS/DBLSTM/DC_WSJ_NOISE'
database_cfg = configparser.ConfigParser()
database_cfg.read(os.path.join(expdir, 'database.conf'))


noise_targets_name = 'testnoisetargets'
noise_targets_dataconf = dict(database_cfg.items(noise_targets_name))

ideal_ratio_name = 'testidealratio'
ideal_ratio_dataconf = dict(database_cfg.items(ideal_ratio_name))


noise_targets_reader = data_reader.DataReader(noise_targets_dataconf)
ideal_ratio_reader = data_reader.DataReader(ideal_ratio_dataconf)

noise_target,_ = noise_targets_reader(5)
ideal_ratio,_=ideal_ratio_reader(5)

print noise_target
print np.shape(noise_target)
print np.sum(ideal_ratio>1.)/np.float(np.prod(np.shape(ideal_ratio)))
