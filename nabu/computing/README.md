# Computing

Nabu has several computing modes. If you train or test a model, you will use
one of these modes to do the computation. You can choose the computation mode
by poining to a configuration file in config/computing in your scripts.

## Standard

The standard computing modes do not use a distributed comuting system (Like
HTCondor or Kubernetes) but rely on the user to determine on what machines they
want to run the scripts.

### Non Distributed

The non distributed mode will run on a single device. This is
the simplest mode of computation. you can choose this mode py pointing to
config/computing/standard/non_distributed.cfg. The script will run on the device
its called from.

## Condor

The Condor compute modes use [HTCondor](https://research.cs.wisc.edu/htcondor/)
to select the machines to run the scripts on instead of relying on the user to do
this. The same computing modes are possible with Condor as he standar compute
modes. The configurations can be found in config/computing/condor. Before you
can start using condor you should create the create_environment.sh script in the
nabu/computing/condor/ directory. This file should create the necesarry
environment variables (like PYTHONPATH) and then execute its arguments. An
example script:

```
#!/bin/sh

#create the necesary environment variables
export PYTHONPATH=/path/to/python/libs

#execute arguments
$@
```

## Torque
Under development
