# Reconstructors

A reconstructor is used to reconstruct signals based on outputs of the network and data in the
database.conf. For example, a mask is determined from the output of the network and is 
multiplied with the original mixture, specified in database.conf. To create a new reconstructor
you should inherit from the general Reconstructor class defined in reconstructor.py and 
overwrite the abstract methods. You should then add it to the factory method in 
reconstructor_factory.py and to the package in \_\_init\_\_.py.

