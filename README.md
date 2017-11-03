**Disclaimer: this code is still under development**

# Nabu-MSSS

Nabu-MSSS (Multi Speaker Source Separation) is an adaptation of Nabu 
(branch 2.0 of Aug 31, 2017, which can be found
[here](https://github.com/vrenkens/nabu)). Nabu is an ASR framework for
end-to-end networks built on top of TensorFlow. Nabu's design focusses on
adaptibility, making it easy for the designer to adjust everything from the
model structure to the way it is trained. 

Because of this adaptibility, many parts of the code of Nabu-MSSS were 
originaly inheritted from Nabu. As a consequence, however, one may still find 
'leftovers' of the original code that do not make much sense for the MSSS
task, eg variable names, code structure, etc. Over time these problems will
be fixed.

## Using Nabu

Nabu works in several stages: data prepation, training and finally testing. 
Each of these stages uses a recipe for a specific model and database. The 
recipe contains configuration files for the all components and defines all
the necesary parameters for the database and the model. You can find more
information on the components in a recipe [here](config/recipes/README.md).

### Data preperation

In the data preperation stage the data is prepared (feature computation,
target normalization etc.) for training and testing. Before running the 
data preperation you should create a database.conf file in the recipe 
directory based on the database.cfg that should already be there, and fill 
in all the paths. In database.conf it is also set wether data shoulbe be 
preprocessed or if it will just be processed on demand.  Should you want 
to modify parameters in the processors, you can modify the config files 
that are pointed to in the database config. You can find more information 
about processors [here](nabu/processing/processors/README.md).

You can run the data prepation with:

```
run data --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing>
```

- recipe: points to the directory containing the recipe you
want to prepare the data for.
- expdir: the path to a directory where you can write to. In this directory all
files will be stored, like the configurations and logs
- computing [default: standard]: the distributed computing software you want to
use. One of standard or condor. standard means that no distributed computing
software is used and the job will run on the machine where nabu is called from.
the condor option uses HTCondor. More information can be found
[here](nabu/computing/README.md). 
**Warning: currently only 'standard' is allowed due to data dependencies. 'condor' computes in parallel**

### Training

In the training stage the model will be trained to minimize a loss function.
During training the model can be evaluated to adjust the learning rate if
necessary. Multiple configuration files in the recipe are used during training:

- model.cfg: model parameters
- trainer.cfg: training parameters
- validation_evaluator.cfg: validation parameters

You can find more information about models
[here](nabu/neuralnetworks/models/README.md), about trainers
[here](nabu/neuralnetworks/trainers/README.md) and about evaluators
[here](nabu/neuralnetworks/evaluators/README.md).

You can run the training with:

```
run train --recipe=/path/to/recipe --expdir=/path/to/expdir --mode=<mode> --computing=<computing>
```

The parameters are the same as the data preperation script (see above) with one
extra parameter; mode (default: non_distributed). Mode is the distribution mode.
This should be one of non_distributed, single_machine or multi_machine.
You can find more information about this [here](nabu/computing/README.md)

**Warning: Currently only 'nondistributed' is allowed for computing**

### Testing

In the testing stage the performance of the model is evaluated on a testing set.
The outputs of the model are used to reconstruct the signal estimates and these
are scored using some scoring metric. To modify the way the model in is evaluated
you can modify the test_evaluator.cfg file in the recipe dir. You can find more
information on evaluators [here](nabu/neuralnetworks/trainers/README.md).

You can run testing with

```
run test --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing>
```

The parameters for this script are similar to the training script (see above).
You should use the same expdir that you used for training the model.


### Parameter search

You can automatically do a parameter search using Nabu. To do this you should
create a sweep file. A sweep file contain blocks of parameters, each block
will change the parameters in the recipe and run a script. A sweep file
looks like this:

```
experiment name 1
confile1 section option value
confile2 section option value
...

experiment name 2
confile1 section option value
confile2 section option value
...

...
```

For example, if you want to try several number of layers and number of units:

```
4layers_1024units
model.cfg encoder num_layers 4
model.cfg encoder num_units 1024

4layers_1024units
model.cfg encoder num_layers 4
model.cfg encoder num_units 2048

5layers_1024units
model.cfg encoder num_layers 5
model.cfg encoder num_units 1024

5layers_1024units
model.cfg encoder num_layers 5
model.cfg encoder num_units 2048
```

The parameter sweep can then be executed as follows:

```
run sweep --command=<command> --sweep=/path/to/sweepfile --expdir=/path/to/exdir <command option>
```

where command can be any of the commands discussed above.

## Designing in Nabu

As mentioned in the beginning Nabu focusses on adaptibility. You can easily
design new models, trainers etc. Most classes used in Nabu have a general class
that defines an interface and common functionality for all children and
a factory that is used to create the necessary class. Look into the respective
README files to see how to implement a new class.

In general, if you want to add your own type of class (like a new model) you
should follow these steps:
- Create a file in the class directory
- Create your child class, let it inherit from the general class and overwrite
the abstract methods
- Add your class to the factory method. In the factory method you give your
class a name, this does not have to be the name of the class. You will use
this name in the configuration file for your model so Nabu knows which class
to use.
- Add your file to the package in \_\_init\_\_.py
- create a configuration file for your class and put it in templates. You
should then add this configuration file in whichever recipe you want to use it
for or create your own recipe using your new class.
