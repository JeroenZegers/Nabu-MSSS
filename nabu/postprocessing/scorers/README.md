# Scorers

A scorer is used to score the reconstruction quality of reconstructed signals, compared to the
original signals. For example, a the SDR between the reconstructions and originals id calculated.
To create a new scorer you should inherit from the general Scorer class defined in scorer.py and 
overwrite the abstract methods. You should then add it to the factory method in 
scorer_factory.py and to the package in \_\_init\_\_.py.

