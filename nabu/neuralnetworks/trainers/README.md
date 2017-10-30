# Trainers

A trainer is used to update the model parameters to minimize some loss function.
To create a new trainer you should inherit from the general MultiTaskTrainer class
defined in multi_task_trainer.py and overwrite the abstract methods. Afterwards yo should
add the trainer to the factory method in trainer_factory.py and the package in
\_\_init\_\_.py

**The Trainer class is no longer supported. Use MultiTaskTrainer and define atleast one task**