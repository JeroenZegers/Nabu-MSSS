**Disclaimer: this code is still under development**

# Nabu-MSSS

Nabu-MSSS (Multi Speaker Source Separation) is an adaptation of [Nabu](https://github.com/vrenkens/nabu)
(branch 2.0 of Aug 31, 2017). Nabu is an ASR framework for
end-to-end networks built on top of TensorFlow. Nabu's design focuses on
adaptability, making it easy for the designer to adjust everything from the
model structure to the way it is trained. 

Code is in Python 2.7 using Tensorflow 1.8.0.

Because of this adaptability, many parts of the code of Nabu-MSSS were 
originally inherited from Nabu. As a consequence, however, one may still find 
'leftovers' of the original code that do not make much sense for the MSSS
task, eg variable names, code structure, etc. 

## Using Nabu

Nabu works in several stages: data preparation, training and finally testing. 
Each of these stages uses a recipe for a specific model and database. The 
recipe contains configuration files for the all components and defines all
the necessary parameters for the database and the model. You can find more
information on the components in a recipe [here](config/recipes/README.md).

### Data preperation

In the data preperation stage the data is prepared (feature computation,
target normalization etc.) for training and testing. Before running the 
data preperation you should create a database.conf file in the recipe 
directory based on the database.cfg that should already be there, and fill 
in all the paths. In database.conf it is also set whether data should be 
preprocessed or if it will just be processed on demand.  Should you want 
to modify parameters in the processors, you can modify the config files 
that are pointed to in the database config. You can find more information 
about processors [here](nabu/processing/processors/README.md).

You can run the data preparation with:

```
run data --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing> --minmemory=<minmemory>
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
- The minimum requested RAM memory in MB when using HTCondor. If not specified, it uses the value in nabu/config/your_computing_type/non_distributed.cfg
 
**Warning: if a data storage directory (see database.conf in recipe) already exists, Nabu-MSSS assumes that data was 
already successfully created and will therefore skip this data section. If it was not successful or you want to redo data
creation, then first remove the storage directory.** 

**Warning: There can be data dependencies between different data section (e.g. mean and variance computed on the 
training set that is applied to validation and test set). Nabu-MSSS does not take this into account and will thus fail. 
A restart is then necessary, but keep in mind the previous warning then.**

### Training

In the training stage the model will be trained to minimize a loss function.
Multiple configuration files in the recipe are used during training:

- model.cfg: model parameters
- trainer.cfg: training parameters
- validation_evaluator.cfg: validation parameters

You can find more information about models
[here](nabu/neuralnetworks/models/README.md), about trainers
[here](nabu/neuralnetworks/trainers/README.md) and about evaluators
[here](nabu/neuralnetworks/evaluators/README.md).

You can run the training with:

```
run train --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing> --minmemory=<minmemory>  
--mincudamemory=<mincudamemory> --resume=<resume> --duplicates=<duplicates> 
--duplicates_ind_offset=<duplicates_ind_offset> --sweep_flag=<sweep_flag> --test_when_finished=<test_when_finished>
```

The parameters are the same as the data preperation script (see above) with extra parameters:
- resume ([default: False]): If resume is set to True, the experiment in expdir, if available, is resumed.
- mincudamemory: The minimum requested Cuda memory in MB when requesting a GPU when using HTCondor. If not specified, 
it uses the value in nabu/config/your_computing_type/non_distributed.cfg
- duplicates ([default: 1]): If duplicates > 1, your_chosen_num_duplicates independent experiments are started. This can be used to
cope with variability in training (e.g. local minima etc.). '_dupl(X)' is appended to the expdir, where X is the 
duplicate index.
- duplicates_ind_offset ([default: 0]):  '_dupl(X+duplicates_ind_offset)' is appended to the expdir
- sweep_flag ([default: False]): Do not set manually, flag used by run_sweep, see further.
- test_when_finished ([default: True]): Not properly implemented yet. It creates a file which contains a "run test"
 command, that should be executed when training has finished. However, currently this command is not executed 
 automatically when training is finished.  

### Testing

In the testing stage the performance of the model is evaluated on a testing set.
The outputs of the model are used to reconstruct the signal estimates and these
are scored using some scoring metric. To modify the way the model in is evaluated
you can modify the test_evaluator.cfg file in the recipe dir. You can find more
information on evaluators [here](nabu/neuralnetworks/trainers/README.md).

You can run testing with

```
run test --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing>  --minmemory=<minmemory>  
--task=<task> --allow_early_testing=<allow_early_testing> --duplicates=<duplicates> 
--duplicates_ind_offset=<duplicates_ind_offset> --sweep_flag=<sweep_flag> --allow_early_testing=<allow_early_testing>
```


You should use the same expdir that you used for training the model.
The parameters for this script are similar to the training script (see above) with extra parameters:
- task ([default: None]): if specified, the test tasks listed in test_evaluater.cfg under 'tasks' will be ignored, and
- allow_early_testing ([default: False]): whether the model can be tested, even if not fully trained yet. The test 
directory will be expdir/test_early_trainstepnumber_of_the_latest_validated_model 

### Parameter sweep

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

4layers_2048units
model.cfg encoder num_layers 4
model.cfg encoder num_units 2048

5layers_1024units
model.cfg encoder num_layers 5
model.cfg encoder num_units 1024
```

The parameter sweep can then be executed as follows:

```
run sweep --recipe=/path/to/recipe --sweep=/path/to/sweepfile --expdir=/path/to/expdir --computing=<computing> 
--minmemory=<minmemory>  --mincudamemory=<mincudamemory> --resume=<resume> --duplicates=<duplicates> 
--duplicates_ind_offset=<duplicates_ind_offset> --test_when_finished=<test_when_finished>
--allow_early_testing=<allow_early_testing> --test_task=<test_task>
```

With the same parameters as before ('task' from run test is renamed to 'test_task').

### Parameter search
Under development.

Alternative to manual parameter sweep. Use Gaussian Processes and acquisition functions to find
the ideal hyperparameters to evaluate. This code was used for the publication Zegers, J., and Van hamme, H. Cnn-lstm 
models for multi-speaker source separation using bayesian hyper parameter optimization. In Interspeech 2019 (2019), 
ISCA, pp. 4589–4593

The parameter sweep can then be executed as follows:

```
run param_search --command=<command> --recipe=/path/to/recipe --hyper_param_conf=/path/to/hyper_param_conf 
--expdir=/path/to/exdir --resume=<resume>
```

With the same parameters as before and:
--hyper_param_conf: the path to the hyper parameter config file.


## Designing in Nabu

As mentioned in the beginning Nabu focuses on adaptability. You can easily
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
- create a configuration file for your class and put it in whichever recipe you
want to use it for.

## Acknowledgments
This work is part of a research project funded by the SB PhD grant of the Research Foundation Flanders 
(FWO) with project number 1S66217N.
Special thanks to the [Facebook AI Research Partnership Program ](https://research.fb.com/facebook-to-accelerate-global-ai-research-with-new-gpu-program-recipients/)
and the [Nvidia GPU Grant Program](https://developer.nvidia.com/academic_gpu_seeding)
for providing additional computational hardware (GPU's) to accelerate testing, debugging and benchmarking 
of the code.

