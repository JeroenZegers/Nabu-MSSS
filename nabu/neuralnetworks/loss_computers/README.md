# Loss computers

A loss computer is used to calculate the loss between the outputs of the model and the labels. It
can be used both for training and evaluation. To create a new loss computer you should inherit 
from the general LossComputer class defined in loss_computer.py and overwrite the abstract 
methods. Afterwards yo should add the loss computer to the factory method in 
loss_computer_factory.py and the package in \_\_init\_\_.py
