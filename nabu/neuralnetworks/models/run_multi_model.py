'''@file run_multi_model.py
script for handling multiple models to form a hybrid model'''

import pdb

def run_multi_model(models, model_nodes, model_links, inputs, inputs_links, 
		    output_names, seq_lengths,is_training):
    '''get the outputs by passing the inputs trought the requested models.
    Model nodes are used to store intermediate results
    
    Args:
	models: dict containing all the models available
	model_nodes: list of all the intermediate model nodes
	model_links: dict containing a model fo each node
	inputs: the inputs to the hybrid model
	inputs_links: dict containing the inputs to the model of the node
	seq_lengths: sequence lengths of the inputs. 
	is_training: whether or not the network is in training mode
  
    Returns:
	outputs: the requested outputs of the hybrid model
    '''
    
    node_tensors = inputs
    for node in model_nodes:
	
	node_inputs = [node_tensors[x] for x in inputs_links[node]]
	node_model = models[model_links[node]]
	#if a model has multiple inputs, only the sequence lenght of the
	#first input will be concidered
	node_seq_length = seq_lengths[inputs_links[node][0]]
	model_output = node_model(
		    inputs=node_inputs,
		    input_seq_length=node_seq_length,
		    is_training=is_training)
	node_tensors[node] = model_output
	seq_lengths[node] = node_seq_length

    outputs = {name: node_tensors[name] for name in output_names}
    
    return outputs

  
def get_variables(models):
    '''get variables of all models
    
    Args:
	models: dict containing all the models available
	
    Returns:
	variables: variables of all models
    '''

    variables=[]
    for model in models.keys():
	variables+=models[model].variables
	
    return variables
	
