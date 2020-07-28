# Models

A model takes multiple inputs and produces multiple outputs.
Multiple models can be combined to create a hybrid model. In run_multi_model.py 
different models are combined, as specified by the user in trainer.cfg. Each task
in *trainer.cfg* specfies input-output behaviour. This is done via model nodes. In *trainer.cfg* the following fields
should be set in each task section to build a (hybrid) model:

- inputs: input_0 input_1
- outputs: output_0 output_1 output_2
- nodes: node_0 node_0 output_0 output_1 output_2
- node_0_inputs: inputs to node_1, can be from *inputs* or from *nodes*
- node_0_model: the model to use for node_0, refers to the model names in model.cfg
- node_1_inputs: ...
- node_1_model: ...
- output_0_inputs: ...
- output_0_model: ...
- output_1_inputs: ...
- output_1_model: ...
- output_2_inputs: ...
- output_2_model: ...
- input_0: name of the train input for input_0, referring to a section in database.conf
- input_1: ...



Example 1: 2 BLSTM layers followed by a linear output layer. Define 2 nodes 'n0' 
and 'output'. Node 'n0' is obtained by using a 'DBLSTM' model, with input 'features'.
Node 'output' is obtained by using a 'Linear' model, with input 'n0'. This is the default hybrid model used in most 
available recipes (e.g. in *config/recipes/papers/ICASSP2018/MERL_DC_2spk*).

Example 2: Same as in example 1 but we expect both mixtures of 2 and 3 speakers. We
want to train the 'main' (DBLSTM in this case) model jointly, but use a different output layer per mixture
type. The recipe for this example can be found in *config/recipes/papers/ICASSP2018/MERL_DC_2spk_and_3spk*.
For the first taks: Node 'n0' is obtained by using a 'DBLSTM' model, with input 
'features' (which refers to the 2spk features). Node 'output' is obtained by using
a 'Linear' model (which will be called 'outlayer_2spk'), with input 'n0'. 
For the second taks: Node 'n0' is obtained by using the same 'DBLSTM' model, with 
input 'features' (which refers to the 3spk features). Node 'output' is obtained by 
using a 'Linear' model (which will be called 'outlayer_3spk), with input 'n0'.

Example 3: Same as in example 1 but we also add oracle ivectors to the input features and
we define 2 output layers (one for Deep Clustering and one for Permutation Invariant
Training). Node 'n0' is obtained by using a 'Concat' model, with inputs 'features' and
'ivecs'. Node 'n1' is obtained by using a 'DBLSTM' model, with input 'n0'. Node 
'output_dc' is obtained by using a 'Linear' model, with input 'n1'. Node 'output_pit' 
is obtained by using a 'Linear' model, with input 'n1'. Notice, that in this set-up the DBLSTM for the Deep Clustering 
and the Permutation Invariant Training will be shared over both tasks.

Example 4: Here we give an example that cannot be found in any of the recipes, but we include it to showcase the wide
variety of architectures you can make using Nabu-MSSS. Lets consider an auto-encoder architecture, where we want to use
the center hidden units for speaker recognition. The encoder model will be automatically shared if you do the following:
- For the auto-encoder task we would use the following nodes: 'encoded'
and 'decoded'. 'encoded_input' would then be set to 'features' and 'encoded_model' would be set to 'encoder', where
'encoder' would then be specified in model.cfg (as for example a 'feedforward' model, where the 'fac_per_layer' option is used to
obtain the encoder architecture. 'fac_per_layer' is the increase factor in nodes per layer. E.g. the number of nodes in
layer l equals 'num_nodes'*('fac_per_layer')^l. Setting 'fac_per_layer' < 1 gives an encoder architecture). We then set
'decoder_input' to 'encoded' and 'decoder_input' to 'decoder', where 'decoder' is again defined in 'model.cfg' (e.g.
using a 'feedforward' model and setting 'fac_per_layer' as the inverse of 'fac_per_layer' in encoder.)
 - For the speaker recognition task we would use the following nodes: 'encoded', 'n0' and 'speaker_logits'. 
 'encoded_input' would again be set to 'features' and 'encoded_model' would again be set to 'encoder'. For 'n0_input' 
 we use 'encoded' and 'n0_model' is set to 'id_net', which can then be specified in *model.cfg*. 'speaker_logits_input'
 is 'n0' and 'speaker_logits_model' is set to 'sr_outlayer' which can then be specified in *model.cfg* to be for instance
 a linear layer with softmax output.



In the *config* directory there a multiple recipes that can serve as an example to
understand how the node-wise modelling works. More information on recipes can be found [here](config/recipes/README.md).