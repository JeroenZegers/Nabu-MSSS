# Evaluators

An evaluator is used to evaluate the performance of the model during training
or at test time. To create a new evaluator you should inherit from the general
Evaluator class defined in evaluator.py and overwrite all the abstract methods.
Afterwards you should add it to the factory method in evaluator_factory.py and
to the package in \_\_init\_\_.py.

