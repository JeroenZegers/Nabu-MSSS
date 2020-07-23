# Recipes

A recipe contains configuration files for training and testing models designed
for a specific database. Nabu-MSSS alleady contains several pre-designed recipes,
but you can design your own recipes with little effort. A recipe contains
the following configurations files (To understand everything, it's best to open an example
recipe, such that things become more clear):

- database.conf: This is the database configuration. For every set of data
(training features, training text, testing features ...) it contains a section
that specifies where to read and write the data in your file system and wich
processors should be used to process the data. Every recipe contains such a 
database configuration, but because the pahs are different for each
person this has to be modified.
- processor configuration files: One or multiple files for the data processors
that are used. The database configuration will point to these processor
configurations. For example, a feature processor specifies the type
of feature, the feature parameters (window length, number of filters, ...) etc.
You can find more information about processors
[here](../../nabu/processing/processors/README.md).
- model.cfg: This is the model configuration. It specifies
the model parameters (number of layers, units ...). The model
configuration contains at least 2 sections:
  - hyper: 
    - model_names: space separated list of the model names
  - model_1: Configuration for the first model
    - architecture: require field to state the used architecture for the model
    - Different (optional) fields like num_layers, num_units, ...
  - model_2
  - ...
You can find more information about models
[here](../../nabu/neuralnetworks/models/README.md).
- trainer.cfg: specifies the trainer parameters (learning rate, nuber of epochs,
...) and the configuration for each train task. The trainer configuration contains at least 2 sections:
    - trainer: gives the general trainer configuration
        - trainer: has to be set to *multi_task*
        - tasks: space separated list of the task names
        - num_epochs
        - valid_frequency
        - ...
    - task_1: Configuration specific to the first traintask.
        - loss_type: The loss type to use, which is defined in loss.cfg
        - inputs: A space separated list of the names that will be used for the inputs
        - outputs: A space separated list of the names that will be used for the outputs
        - nodes: A space separated list of the names that will be used for the nodes to build the models.
        - node_1_inputs: inputs to node_1, can be from *inputs* or from *nodes*
        - node_1_model: the model to use fornode_1, refers to the model names in model.cfg
        - node_2_inputs: ...
        - node_2_model: ...
        - input_1: name of the train input for input_1, referring to a section in database.conf
        - input_2: ...
        - targets:  space separated list of the names that will be used for the targets
        - target_1: name of the train target for target_1, referring to a section in database.conf
        - target_2:
        
        An example for a DeepClustering task:
        - loss_type = deepclustering
        - inputs = features
        - outputs = bin_emb
        - nodes = n0 bin_emb
        - n0_model = main
        - n0_inputs = features
        - bin_emb_model = outlayer
        - bin_emb_inputs = n0
        - features = trainspec
        - targets = binary_targets usedbins 
        - binary_targets = traintargets 
        - usedbins = trainusedbins
    - task_2
    - ...
 
You can find more information about trainers
[here](../../nabu/neuralnetworks/trainers/README.md).
- validation_evaluator.cfg: Similar to trainer.cfg. The validator configuration contains at least 2 sections:
    - evaluator:
        - evaluator: should always be set to *task_loss_evaluator*
        - requested_utts: the number of utterances to use for validation
        - batch_size = The validation batch size
        - tasks_excluded_for_val: an optional, space separated list of tasks defined in trainer.cfg, that should not be
        used for validation-based early stopping
        - ...
    - task_1: where the task_name was defined in trainer.cfg. Similar as for trainer.cfg, but the references to inputs
    and targets should refer to validation data (instead of train data).
    - task_2: ...
    - ...
    
You can find more information about evaluators
[here](../../nabu/neuralnetworks/evaluators/README.md)
- test_evaluator.cfg: This is the configuartion for the evaluator to be used at
test time (see validation_evaluator.cfg)
- reconstructor.cfg: During testing, this configuration is used to reconstruct the speech estimates. It contains
atleast 1 section:
    - task_1: where the task name was defined in evaluator.cfg
        - reconstruct_type
        - ...

You can find more information about reconstructors 
        [here](../../nabu/postprocessing/reconstructors/README.md)
- scorer.cfg: after reconstructing, this configuration is used to score the separation quality of the speech estimates. 
It contains atleast 1 section:
    - task_1: where the task name was defined in evaluator.cfg
        - score_type
        - ...

You can find more information about scorers 
        [here](../../nabu/postprocessing/scorers/README.md)
- loss.cfg: For each loss_type, referenced to in trainer.cfg, validation_evalutor.cfg and/or test_evaluator.cfg a
section has to be made
    - loss_1
        - loss_type
        - ...
    - loss_2
    - ...
You can find more information about losses 
        [here](../../nabu/neuralnetworks/loss_computers/README.md)
        
To create your own recipe, simply create a directory containing all of the
mentioned configuation files. You can find template configuations in
config/templates.

In the directories *papers* and *SS*, one can find many example recipes.