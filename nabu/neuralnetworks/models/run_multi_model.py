'''@file run_multi_model.py
script for handling multiple models'''

import pdb

def run_multi_model(models, model_paths,inputs,seq_length,is_training):
    '''get the output by running multiple models in turn
    
    Args:
	models: dict containing all the models available
	model_paths: list of models that should be used and in which order
	inputs: the inputs to the first model
	seq_length: seq_length for the inputs
	is_training: whether or not the network is in training mode
  
    Returns:
	model_output: the output of the last model
    '''
    
    model_inputs=inputs
    for model_path in model_paths:
	model_output = models[model_path](
		    inputs=model_inputs,
		    input_seq_length=seq_length,
		    is_training=is_training)
	#only works for 1 input
	model_inputs[model_inputs.keys()[0]] = model_output
	
	
    return model_output
  
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
	