# Models

A model takes multiple inputs and produces multiple outputs.
Multiple models can be combined to create a hybrid model. In run_multi_model.py 
different models are combined, as specified by the user in trainer.cfg. Each task
in trainer.cfg specfies input-output behaviour. This is done via model nodes.


Example 1: 2 BLSTM layers followed by a linear output layer. Define 2 nodes 'n0' 
and 'output'. Node 'n0' is obtained by using a 'DBLSTM' model, with input 'features'.
Node 'output' is obtained by using a 'Linear' model, with input 'n0'

Example 2: Same as in example 1 but we also add ivectors to the input features and
we define 2 output layers (one for Deep Clustering and one for Permutation Invariant
Training). Node 'n0' is obtained by using a 'Concat' model, with inputs 'features' and
'ivecs'. Node 'n1' is obtained by using a 'DBLSTM' model, with input 'no'. Node 
'output_dc' is obtained by using a 'Linear' model, with input 'n1'. Node 'output_pit' 
is obtained by using a 'Linear' model, with input 'n1'.

Example 3: Same as in example 1 but we expect both mixtures of 2 and 3 speakers. We
want to train the 'main' model jointly, but use a different output layer per mixture
type. 
For the first taks: Node 'n0' is obtained by using a 'DBLSTM' model, with input 
'features' (which refers to the 2spk features). Node 'output' is obtained by using
a 'Linear' model (which will be called 'outlayer_2spk), with input 'n0'. 
For the second taks: Node 'n0' is obtained by using the same 'DBLSTM' model, with 
input 'features' (which refers to the 3spk features). Node 'output' is obtained by 
using a 'Linear' model (which will be called 'outlayer_3spk), with input 'n0'.

In the config directory there a multiple recipes that can serve as an example to
understand how the node-wise modelling works.