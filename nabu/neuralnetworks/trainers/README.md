# Trainers

A trainer is used to update the model parameters to minimize some loss function.
To create a new trainer you should inherit from the general Trainer class
defined in trainer.py and overwrite the abstract methods. Afterwards yo should
add the trainer to the factory method in trainer_factory.py and the package in
\_\_init\_\_.py

Added a trainer for multi task learning. Currently only for tasks that use the 
same model. the global loss is just the mean of the losses of the tasks. Will
generalize to general multi task learning (e.g. where part of the model is 
shared, adverserial networks, ...)