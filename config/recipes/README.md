# Recipes

A recipe contains configuration files for training and testing models designed
for a specific database. Nabu-MSSS alleady contains several pre-designed recipes,
but you can design your own recipes with little effort. A recipe contains
the following configurations files:

- database.conf: This is the database configuration. For every set of data
(training features, training targets, testing features ...) it contains a section
that specifies where to read and write the data in your file system and wich
- processor configuration files: One or multiple files for the data processors
that are used. The database configuration will point to these processor
confiurations. For example, a feature processor specifies the type
of feature, the feature parameters (window length, number of filters, ...) etc.
You can find more information about processors
[here](../../nabu/processing/processors/README.md).
- model.cfg: This is the model configuration. A model is the agglomeration of
multiple submodels. For each submodel, the user can define the model parameters 
(number of layers, units ...).
You can find more information about models
[here](../../nabu/neuralnetworks/models/README.md).
- trainer.cfg: specifies the trainer parameters (learning rate, nuber of epochs,
...). It also specifies which tasks will be trained. For each task inputs, outputs
and targets are defined, as well as wich submodels should be used to obtain the 
desired input-output behaviour. You can find more information about trainers
[here](../../nabu/neuralnetworks/trainers/README.md).
- validation_evaluator.cfg: specifies the validator type and parameters,
that will be used during training. It also specifies which tasks will be validated. 
For each task inputs, outputs and targets are defined, as well as wich submodels 
should be used to retrieve the outputs from the inputs. validation evaluator is 
used during training to measure the performance at fixed intervals and adjusts the
learning rate if necesarry. You can find more information about evaluators
[here](../../nabu/neuralnetworks/evaluators/README.md)
- test_evaluator.cfg: This is the configuartion for the evaluator to be used at
test time (see validation_evaluator.cfg)
- reconstructor.cfg: specifies how the (speech) signals should be reconstructed
from outputs of the model
- scorer.cfg: specfies how the reconstructed signals should be scored (e.g. using
SDR, SIR, ...)

To create your own recipe, simply create a directory containing all of the
mentioned configuation files. You can find template configuations in
config/templates.
