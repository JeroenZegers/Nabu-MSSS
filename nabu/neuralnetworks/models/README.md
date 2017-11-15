# Models

A model takes multiple inputs and produces multiple outputs.
Multiple models can be combined to create a hybrid model. In run_multi_model.py 
different models are combined, as specified by the user.

Example 1: 2 BLSTM layers, followed by 2 feedforward layers

Example 2: A couple of shared layers and a separate output layer per scenario


**warning: currently only one input per output is allowed**
