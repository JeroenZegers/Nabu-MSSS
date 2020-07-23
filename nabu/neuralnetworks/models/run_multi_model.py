"""@file run_multi_model.py
script for handling multiple models to form a hybrid model"""
from nabu.neuralnetworks.models.framer import Framer, DeframerSelect
import pdb


def run_multi_model(
		models, model_nodes, model_links, inputs, inputs_links, nodes_output_names, output_names, seq_lengths,
		is_training):
	"""get the outputs by passing the inputs trough the requested models.
	Model nodes are used to store intermediate results

	Args:
	models: dict containing all the models available
	model_nodes: list of all the intermediate model nodes
	model_links: dict containing a model fo each node
	inputs: the inputs to the hybrid model
	inputs_links: dict containing the inputs to the model of the node
	nodes_output_names: dict containing the output names of the node's model. It can thus be allowed to make a
		'multi-node' node. Typically only one output name is expected, which is the name of the node itself.
	seq_lengths: sequence lengths of the inputs.
	is_training: whether or not the network is in training mode

	Returns:
	outputs: the requested outputs of the hybrid model
	used_models: the models used to form the hybrid model
	"""

	node_tensors = inputs
	used_models = dict()
	for node in model_nodes:
		node_inputs = [node_tensors[x] for x in inputs_links[node]]
		node_model_name = model_links[node]
		node_model = models[node_model_name]
		used_models[node_model_name] = node_model
		node_output_names = nodes_output_names[node]

		# else condition is also for legacy models.
		if hasattr(node_model, 'num_outputs'):
			node_num_outputs = node_model.num_outputs
		else:
			node_num_outputs = 1

		if isinstance(node_model, Framer):
			# exceptional case
			# batch_size = node_inputs[0].get_shape()[0]
			# T = tf.shape(node_inputs)[1]
			node_seq_length_before_framer = seq_lengths[inputs_links[node][0]]
			# node_seq_length = tf.ones(batch_size*T,dtype=tf.int32)*node_model.frame_length
			node_seq_length = None
		elif isinstance(node_model, DeframerSelect):
			# exceptional case
			node_seq_length = node_seq_length_before_framer
		else:
			# if a model has multiple inputs, only the sequence length of the
			# first input will be considered
			node_seq_length = seq_lengths[inputs_links[node][0]]

		model_outputs = node_model(
				inputs=node_inputs,
				input_seq_length=node_seq_length,
				is_training=is_training)
		if node_num_outputs == 1:
			node_tensors[node_output_names] = model_outputs
			seq_lengths[node_output_names] = node_seq_length
		else:
			node_tensors.update({node_output_names[ind]: model_outputs[ind] for ind in range(node_num_outputs)})
			seq_lengths.update({node_output_names[ind]: node_seq_length for ind in range(node_num_outputs)})

	outputs = {name: node_tensors[name] for name in output_names}

	return outputs, used_models


def get_variables(models):
	"""get variables of all models

	Args:
	models: dict containing all the models available

	Returns:
	variables: variables of all models
	"""

	variables = []
	for model in models.keys():
		variables += models[model].variables

	return variables
