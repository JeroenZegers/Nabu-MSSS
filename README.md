**Disclaimer: this code is still under development**

# Nabu-MSSS

Nabu-MSSS contains for Multi Speaker Source Separation with neural networks, build with TensorFlow.
Nabu-MSSS is an adaptation of [Nabu](https://github.com/vrenkens/nabu)
(branch 2.0 of Aug 31, 2017). The original Nabu is an ASR framework for
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
already successfully created before and will therefore skip this data section. If it was not successful or you want to redo data
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

## Step-by-step procedure to create your first model
We'll be using the *config/recipes/papers/ICASSP2018/MERL_DC_2spk* recipe to build your first model.

### Prior stuff
- Put the Nabu directory in your python path
- Install TensorFlow v1.8.0
- Use Python2.6

### Data creation
- Create datafiles for the train set, the validation set and the test set. If you have a dataset where only the mixtures
are available, but the target signals, see section *Separating mixtures without available targets*.
    - the mix_wav.scp file which is of the form
        ```
        mix_1_name path_to_mix1.wav
        mix_2_name path_to_mix2.wav
        mix_3_name path_to_mix3.wav
        ...
        ```
    - the allS_wav.scp file which is of the form
        ```
        mix_1_name path_to_speech_of_speaker1_for_mix1.wav path_to_speech_of_speaker2_for_mix1.wav
        mix_2_name path_to_speech_of_speaker1_for_mix2.wav path_to_speech_of_speaker2_for_mix2.wav
        mix_3_name path_to_speech_of_speaker1_for_mix3.wav path_to_speech_of_speaker2_for_mix3.wav
        ...
        ```
    - the utt_spkinfo file which is of the form
        ```
        mix_1_name path_to_mix1.wav id_of_speaker1_for_mix1 gender_of_speaker1 (M or F) id_of_speaker2_for_mix1 gender_of_speaker2
        mix_2_name path_to_mix2.wav id_of_speaker1_for_mix2 gender_of_speaker1 (M or F) id_of_speaker2_for_mix2 gender_of_speaker2
        mix_3_name path_to_mix3.wav id_of_speaker1_for_mix3 gender_of_speaker1 (M or F) id_of_speaker2_for_mix3 gender_of_speaker2
        ...
        ```
- Modify the *datafiles* fields in *config/recipes/papers/ICASSP2018/MERL_DC_2spk/database.conf* such that they point to
 your datafiles.
- In the same *database.conf*, change */esat/spchtemp/scratch/jzegers/dataforTF/MERL_segmented/* in the *store_dir* 
fields to *path_to_your_datastore_dir* 
    
- Run the following:
```
run data --computing=condor --expdir=your_nabu_experiment_directory/MERL_DC_2spk --recipe=config/recipes/papers/ICASSP2018/MERL_DC_2spk
```
The jobs for testspec en devspec should fail, as they rely on trainspec which is not yet finished. Once trainspec is
finished, remove *path_to_your_datastore_dir/features/dev* and *path_to_your_datastore_dir/features/test* and rerun the above command.
If data creation was successful the following files should exist 
*path_to_your_datastore_dir/{features,targets,usedbins}/{tr,dev,test}/{100,full}/pointers.scp*

### Model training
- Once all data has been prepared, you can run the following:
```
run train --test_when_finished=False --computing=condor --expdir=your_nabu_experiment_directory/MERL_DC_2spk --recipe=config/recipes/papers/ICASSP2018/MERL_DC_2spk
```
### Model evaluation
- Once the model has finished training, you can run the following:
```
run test --computing=condor --expdir=your_nabu_experiment_directory/MERL_DC_2spk --recipe=config/recipes/papers/ICASSP2018/MERL_DC_2spk
```
You can add the option *--allow_early_testing=True* to the above command to start testing before the model has finished
trainig (Note however, that typically in *test_evaluator.cfg* the requested segment_lenth is set to *full*, thus the 
 model should have been atleast validated once during training on *full* before this option will work).
 
The separation score can then be observed in *your_nabu_experiment_directory/MERL_DC_2spk/test/outputs/main_task_2spk.out*
and in *your_nabu_experiment_directory/MERL_DC_2spk/test/results_task_2spk_sdr_summary.json*

### Separating mixtures without available targets
Should you simply want to separate some mixtures, but do not have the original clean speaker signals available, you can 
do the following:
- Create the mix_wav.scp file as stated in the *Data creation* section.
- In database.conf:
    - Duplicate the *testspec*, *testusedbins* and *testorgmix* sections and change the names to *your_dataset_namespec*
     ,*your_dataset_nameusedbins* and *your_dataset_nameorgmix*.
    - Modify the *datafiles* and *store_dir* fields in *your_dataset_namespec* and *your_dataset_nameusedbins* to your 
    own paths as explained in the *Data creation* section.
- In test_evaluator.conf:
    - Duplicate the *task_2spk* section and change the name to *task_your_dataset*
    - In the *evaluator* section, modify the *tasks* field to *tasks = task_your_dataset*
    - In the new *task_your_dataset* section:
        - Modify the *requested_utts* to your dataset size (and possibly the *batch_size* as well
        - Modify the *loss_type* field to *loss_type = dummy*
        - Modify the *features* field to *features = your_dataset_namespec*
        - Modify the *targets* field to *targets = dummy_targets*
        - Modify the *binary_targets* field to *dummy_targets = your_dataset_namespec*
        - Remove the *usedbins* field
- In reconstructor.conf
    - Duplicate the *task_2spk* section and change the name to *task_your_dataset*
    - Modify the *org_mix* field to *org_mix = your_dataset_nameorgmix*
    - Modify the *usedbins* field to *usedbins = your_dataset_nameusedbins*
- In loss.conf: add a new section *dummy* and in this section add the following field *loss_type = dummy*.
- Run the command test command:
    ```
    run test --computing=condor --expdir=your_nabu_experiment_directory/MERL_DC_2spk --recipe=config/recipes/papers/ICASSP2018/MERL_DC_2spk
    ```
- At some point the command will return an error because it tries to score the reconstructions, but it does not have the
clean target signals. However, the reconstructions (single speaker signal estimates) should be available in 
*your_nabu_experiment_directory/MERL_DC_2spk/test/reconstructions/task_your_dataset*. The error, which can be ignored,
would look something like this
    ```
    Traceback (most recent call last):
      File "/users/spraak/jzegers/Documents/Nabu-SS_from_scratch/nabu/scripts/prepare_test.py", line 258, in <module>
        tf.app.run()
      File "/users/spraak/spch/prog/spch/tensorflow-1.8.0/lib/python2.7/site-packages/tensorflow/python/platform/app.py", line 126, in run
        _sys.exit(main(argv))
      File "/users/spraak/jzegers/Documents/Nabu-SS_from_scratch/nabu/scripts/prepare_test.py", line 229, in main
        test(expdir=test_expdir_run, test_model_checkpoint=test_model_checkpoint, task=task)
      File "/users/spraak/jzegers/Documents/Nabu-SS_from_scratch/nabu/scripts/test.py", line 285, in test
        task_scorer_cfg = dict(scorer_cfg.items(scorer_name))
      File "/users/spraak/spch/prog/packages/python/lib/python2.7/ConfigParser.py", line 642, in items
        raise NoSectionError(section)
    ConfigParser.NoSectionError: No section: 'task_your_dataset'
    ```

## Acknowledgments
This work is part of a research project funded by the SB PhD grant of the Research Foundation Flanders 
(FWO) with project number 1S66217N.
Special thanks to the [Facebook AI Research Partnership Program ](https://research.fb.com/facebook-to-accelerate-global-ai-research-with-new-gpu-program-recipients/)
and the [Nvidia GPU Grant Program](https://developer.nvidia.com/academic_gpu_seeding)
for providing additional computational hardware (GPU's) to accelerate testing, debugging and benchmarking 
of the code.

