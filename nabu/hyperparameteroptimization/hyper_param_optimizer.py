import optimizer_plots as plots

import cPickle as pickle
import copy
import json
import math
import os
import shutil
import time
import warnings
import sys
import numpy as np
import skopt
from scipy.optimize import fmin_l_bfgs_b
from six.moves import configparser
from sklearn import clone
from sklearn.externals.joblib import Parallel, delayed
from skopt.acquisition import _gaussian_acquisition, gaussian_acquisition_1D
from skopt.space import space as skopt_space
from skopt.space import Space
from skopt.utils import is_2Dlistlike, is_listlike, create_result, normalize_dimensions

from skopt.learning.gaussian_process.kernels import ConstantKernel
from skopt.learning.gaussian_process.kernels import HammingKernel
from skopt.learning.gaussian_process.kernels import Matern

from estimators import BoundedGaussianProcessRegressor
from utils import check_parameter_count, check_parameter_count_for_sample, distance, partial_dependence_valid_samples


class HyperParamOptimizer(skopt.Optimizer):
	def __init__(self, hyper_param_conf, command, expdir, exp_recipe_dir, recipe, computing, exp_proposal_watch_dir=None):
		base_estimator = 'GP'

		self.hyper_param_conf = hyper_param_conf
		self.command = command
		self.expdir = expdir
		self.exp_recipe_dir = exp_recipe_dir
		self.recipe = recipe
		self.computing = computing

		# read the hyper parameter file
		hyper_param_cfg = configparser.ConfigParser()
		hyper_param_cfg.read(hyper_param_conf)

		hyper_info = dict(hyper_param_cfg.items('info'))
		self.hyper_param_names = hyper_info['hyper_params'].split(' ')
		self.num_iters = int(hyper_info['num_iters'])
		self.n_initial_points = int(hyper_info['n_initial_points'])
		self.n_initial_points_to_start = int(hyper_info['n_initial_points_to_start'])
		self.max_parallel_jobs = int(hyper_info['max_parallel_jobs'])
		self.selected_segment_length = hyper_info['segment_length']
		self.selected_task = hyper_info['task']

		if 'adapt_hyper_param' in hyper_info:
			self.adapt_param = {
				'param_name': hyper_info['adapt_hyper_param'], 'param_thr': int(hyper_info['param_thr']),
				'par_cnt_scheme': hyper_info['par_cnt_scheme']}
		else:
			self.adapt_param = None

		hyper_param_dict = dict()
		skopt_dims = []
		for par_name in self.hyper_param_names:
			par_dict = dict(hyper_param_cfg.items(par_name))
			par_type = par_dict['type']
			if par_type == 'Integer':
				skopt_dim = skopt_space.Integer(
					low=int(par_dict['min']), high=int(par_dict['max']), name=par_name)

			elif par_type == 'Real':
				skopt_dim = skopt_space.Real(
					low=float(par_dict['min']), high=float(par_dict['max']), name=par_name)

			elif par_type == 'Categorical':
				skopt_dim = skopt_space.Categorical(categories=par_dict['categories'].split(' '), name=par_name)

			else:
				raise ValueError('Type %s is not a valid parameter type' % par_type)

			hyper_param_dict[par_name] = par_dict
			skopt_dims.append(skopt_dim)

		self.hyper_param_dict = hyper_param_dict
		self.skopt_dims = skopt_dims

		self.last_result = None
		# self.all_results = []

		self.start_new_run_flag = True
		self.iter_ind = 0
		self.watch_list = dict()
		self.all_dim_values = []
		self.all_losses = dict()
		self.n_job_running = 0
		self.n_initial_points_started = 0
		self.n_unsuitable_points_for_estimator = 0
		self.max_n_unsuitable_points_for_estimator = 10000
		self.unsuitable_runs = []
		self.lost_runs = []

		self.exp_proposal_watch_dir = exp_proposal_watch_dir
		self.use_proposal_run = False
		self.proposed_loss_runs = []

		# only 0.25% of the point sample in the hyper space are wanted (since they lead to rougly the wanted amount of
		# trainable parameters)
		self.acq_optimizer_kwargs = {'n_points': 4000000}
		if 'debug' in expdir:
			self.acq_optimizer_kwargs = {'n_points': 40000}

		if base_estimator == 'boundedGP':
			# Make own estimator based on Gaussian Process Regressor.
			if skopt_dims is not None:
				space = Space(skopt_dims)
				space = Space(normalize_dimensions(space.dimensions))
				n_dims = space.transformed_n_dims
				is_cat = space.is_categorical

			else:
				raise ValueError("Expected a Space instance, not None.")

			cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
			# only special if *all* dimensions are categorical
			if is_cat:
				other_kernel = HammingKernel(length_scale=np.ones(n_dims))
			else:
				other_kernel = Matern(
					length_scale=np.ones(n_dims),
					length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)

			base_estimator = BoundedGaussianProcessRegressor(
				space, self.hyper_param_names, self.adapt_param, kernel=cov_amplitude * other_kernel, normalize_y=True,
				noise="gaussian", n_restarts_optimizer=2)

		super(HyperParamOptimizer, self).__init__(
			skopt_dims, base_estimator=base_estimator, n_initial_points=self.n_initial_points,
			acq_optimizer_kwargs=self.acq_optimizer_kwargs)

	def __call__(self):

		while (self.iter_ind - self.n_unsuitable_points_for_estimator) < self.num_iters or len(self.watch_list) > 0:
			# check if the user proposed hyper parameter values (and potentially a corresponding validation loss)
			if self.exp_proposal_watch_dir is not None:
				self.use_proposal_run = self.watch_proposal_dir()

			if False and self.start_new_run_flag or self.use_proposal_run:
				# start (a) new run(s) if allowed
				self.start_new_runs()
				self.checkpoint()
			else:
				time.sleep(0.5)

				# check whether any run has finished (if a run has not finished and check_jobs is True, check whether
				# the job is still present in Condor)
				self.check_watch_files()

			# check whether a new run should be stared
			if \
					(
							self.n_job_running < self.max_parallel_jobs or
							self.n_initial_points_started < self.n_initial_points_to_start
					) and self.iter_ind < (self.num_iters + self.n_unsuitable_points_for_estimator):
				self.start_new_run_flag = True
			else:
				self.start_new_run_flag = False

	def watch_proposal_dir(self):
		files_in_dir = [
			f for f in os.listdir(self.exp_proposal_watch_dir)
			if os.path.isfile(os.path.join(self.exp_proposal_watch_dir, f))]

		all_x_vals = []
		only_pars = [True] * len(files_in_dir)
		for file_ind, file_in_dir in enumerate(files_in_dir):
			full_file = os.path.join(self.exp_proposal_watch_dir, file_in_dir)
			with open(full_file, 'r') as fid:
				tmp = fid.read().split('\n')
				x_str_vals = tmp[0].split(',')
				if len(x_str_vals) != len(self.skopt_dims):
					raise ValueError(
						'%d values were proposed for %d hyper parameters in %s' %
						(len(x_str_vals), len(self.skopt_dims), file_in_dir))
				x_vals = []
				for ind, dim in enumerate(self.skopt_dims):
					if isinstance(dim, skopt_space.Integer):
						x_vals.append(int(x_str_vals[ind]))
					elif isinstance(dim, skopt_space.Real):
						x_vals.append(float(x_str_vals[ind]))
					elif isinstance(dim, skopt_space.Categorical):
						x_vals.append(x_str_vals[ind])
					else:
						raise ValueError('Unexpected value type')
				all_x_vals.append(x_vals)

				# if a loss is found as well, tell the estimator
				if len(tmp) > 1 and tmp[1] != '':
					y_val = float(tmp[1])
					only_pars[file_ind] = False

					self.iter_ind += 1
					self.n_initial_points_started += 1
					self.all_dim_values.append(x_vals)
					self.proposed_loss_runs.append(self.iter_ind)
					self.process_loss(y_val, ind)

					self.checkpoint()

					os.remove(full_file)

		# run one proposed hyper parameter at a time
		tmp = [ind for ind, only_par in enumerate(only_pars) if only_par]
		if len(tmp) > 0:
			chosen_run = tmp[0]
			chosen_x_vals = all_x_vals[chosen_run]
			self.proposal_run_vals = chosen_x_vals
			use_proposal_run = True
			full_file = os.path.join(self.exp_proposal_watch_dir, files_in_dir[chosen_run])
			os.remove(full_file)
		else:
			use_proposal_run = False

		return use_proposal_run

	def start_new_runs(self):

		if self.use_proposal_run:
			# run the proposed values by the user
			dim_values, fixed_suitable_values = self.adapt_hyper_param(self.proposal_run_vals)
			self.proposal_run_vals = []
			if not fixed_suitable_values:
				print 'Proposed values are not allowed! Ignoring them'
				return
			self.n_initial_points_started += 1
			self.use_proposal_run = False

			self.start_new_run(dim_values)

		elif self.n_initial_points_started < self.n_initial_points_to_start:
			# for the first n_initial_points_to_start, actively look for valid hyper param values, using an adaptation
			# technique.

			fixed_suitable_values = False
			while not fixed_suitable_values:
				dim_values = self.ask()
				if self.adapt_param is None:
					fixed_suitable_values = True
				else:
					dim_values, fixed_suitable_values = self.adapt_hyper_param(dim_values)

			self.n_initial_points_started += 1

			self.start_new_run(dim_values)

		else:
			# use the estimator to ask for proposed hyper parameters
			multi_dim_values = self.ask(n_points=self.max_parallel_jobs - self.n_job_running, strategy='cl_mean')
			param_thr = self.adapt_param['param_thr']
			par_cnt_scheme = self.adapt_param['par_cnt_scheme']
			multi_unsuitable_values = []

			for dim_values in multi_dim_values:
				suitable_values, par_cnt_dict = check_parameter_count_for_sample(
					dim_values, self.hyper_param_names, param_thr, par_cnt_scheme)
				if not suitable_values:
					if self.n_unsuitable_points_for_estimator < self.max_n_unsuitable_points_for_estimator:
						multi_unsuitable_values.append(dim_values)
					else:
						# do nothing. Try again for a suitable point or wait for a valid point to be returned to update the
						# estimator (whichever comes first).
						# TODO: do adapt_hyper_param, otherwise we might never get out of here. And possibly do
						# ask(n_points=100) to get more points
						print 'Found too many unsuitable values, doing nothing.'
				else:
					self.start_new_run(dim_values)

			if multi_unsuitable_values:
				n_unsuitable_values = len(multi_unsuitable_values)
				print 'Using %d unsuitable values to lie about' % n_unsuitable_values
				# after telling the estimator these (useless) points it will decrease self._n_initial_points by 1 every
				# time, but we don't want to consider these (useless) points as a initial points
				self._n_initial_points += n_unsuitable_values
				# tell the estimator an artificial, high loss for the unsuitable hyper param values.
				artificial_losses = [0.175] * n_unsuitable_values
				prev_time = time.time()
				self.tell(multi_unsuitable_values, artificial_losses, fit=True)
				print (time.time()-prev_time)
				# increase the count for unsuitable points given to the estimator
				self.n_unsuitable_points_for_estimator += n_unsuitable_values
				new_iter_inds = range(self.iter_ind, self.iter_ind+n_unsuitable_values)
				self.unsuitable_runs.extend(new_iter_inds)
				self.iter_ind += n_unsuitable_values
				self.all_dim_values.extend(multi_unsuitable_values)
				# if np.mod(self.n_unsuitable_points_for_estimator, 100) == 0:
				# 	print \
				# 		'Hit unsuitable point number %d.' % self.n_unsuitable_points_for_estimator

		# if self.iter_ind == 0:
		# 	# check for default values
		# 	for par_ind, par_name in enumerate(self.hyper_param_names):
		# 		if 'default' in self.hyper_param_dict[par_name]:
		# 			if self.hyper_param_dict[par_name]['type'] == 'Integer':
		# 				dim_values[par_ind] = int(self.hyper_param_dict[par_name]['default'])
		# 			elif self.hyper_param_dict[par_name]['type'] == 'Real':
		# 				dim_values[par_ind] = float(self.hyper_param_dict[par_name]['default'])
		# 			elif self.hyper_param_dict[par_name]['type'] == 'Categorical':
		# 				dim_values[par_ind] = self.hyper_param_dict[par_name]['default']
		# 			else:
		# 				raise ValueError(
		# 					'Type %s is not a valid parameter type' % self.hyper_param_dict[par_name]['type'])

		return

	def start_new_run(self, dim_values):
		it_expname = 'run_' + str(self.iter_ind)
		it_expdir = os.path.join(self.expdir, it_expname)
		it_exp_recipe_dir = os.path.join(self.exp_recipe_dir, it_expname)

		if os.path.isdir(it_expdir):
			print 'WARNING: %s is already a directory!' % it_expdir
			shutil.rmtree(it_expdir)
		os.makedirs(it_expdir)
		if os.path.isdir(it_exp_recipe_dir):
			print 'WARNING: %s is already a directory!' % it_exp_recipe_dir
			shutil.rmtree(it_exp_recipe_dir)
		shutil.copytree(self.recipe, it_exp_recipe_dir)

		# adapt the config files according to dim_values
		self.prepare_configs(config_dir=it_exp_recipe_dir, requested_values=dim_values)
		print '*** STARTING NEW MODEL ***'
		print 'Model %d will use values:' % self.iter_ind
		print dim_values

		# compare the new model with previous models
		all_distances = []
		all_runs_inds = []
		for ref_run_ind, ref_values in enumerate(self.all_dim_values):
			if ref_run_ind not in self.unsuitable_runs:
				dist = distance(self.space, dim_values, ref_values)
				all_distances.append(dist)
				all_runs_inds.append(ref_run_ind)
		tmp = np.argsort(all_distances)
		sorted_distances = [all_distances[tmpi] for tmpi in tmp]
		sorted_runs_inds = [all_runs_inds[tmpi] for tmpi in tmp]
		closest_run = sorted_runs_inds[0]
		smallest_distance = sorted_distances[0]
		sorted_runs_with_loss_inds = [ind for ind in sorted_runs_inds if ind in self.all_losses.keys()]
		sorted_distances_with_loss = \
			[sorted_distances[tmpi] for tmpi, ind in enumerate(sorted_runs_inds) if ind in self.all_losses.keys()]
		closest_run_with_loss = sorted_runs_with_loss_inds[0]
		smallest_distance_with_loss = sorted_distances_with_loss[0]

		print \
			'Closest previous model is run_%d (distance: %f, loss=%f) with values:' \
			% (closest_run_with_loss, smallest_distance_with_loss, self.all_losses[closest_run_with_loss])
		print self.all_dim_values[closest_run_with_loss]
		if closest_run_with_loss != closest_run:
			print \
				'Closest previous model without evaluated loss is run_%d (distance: %f) with values:' \
				% (closest_run, smallest_distance)
			print self.all_dim_values[closest_run]

		# train and validate a model with the above config files
		job_string = 'run %s --expdir=%s --recipe=%s --computing=%s --sweep_flag=%s' % (
			self.command, it_expdir, it_exp_recipe_dir, self.computing, True)

		file_to_watch = os.path.join(it_expdir, 'val_sum.json')
		self.watch_list[self.iter_ind] = file_to_watch
		self.all_dim_values.append(dim_values)
		self.iter_ind += 1
		self.n_job_running += 1

	def check_watch_files(self):
		found_losses = dict()
		for ind, watch_file in self.watch_list.iteritems():
			if os.path.isfile(watch_file):
				with open(watch_file, 'r') as fid:
					val_sum = json.load(fid)

					if self.selected_segment_length not in val_sum:
						raise ValueError(
							'did not find segment length %s in "val_sum.json"' % self.selected_segment_length)
					loss = val_sum[self.selected_segment_length]

					if self.selected_task not in loss:
						raise ValueError('did not find task %s in "val_sum.json"' % self.selected_task)
					loss = loss[self.selected_task]

				found_losses[ind] = loss

		n_found_losses = len(found_losses)
		if n_found_losses > 0:

			self.n_job_running -= n_found_losses

			for ind in found_losses:
				del self.watch_list[ind]

			self.process_losses(found_losses)

			self.checkpoint()

	def process_losses(self, losses):

		for run_ind, loss in losses.iteritems():

			print 'Found loss %.3f for model %d' % (loss, run_ind)
			if math.isnan(loss):
				loss = 0.17
				print 'Found loss = NaN. Changing to loss = 0.17'
			if loss > 0.17:
				loss = 0.17
				print 'Found high loss. Changing to loss = 0.17'
			self.all_losses[run_ind] = loss
			losses[run_ind] = loss

		dim_values_of_losses = [self.all_dim_values[ind] for ind in losses]
		found_losses = [losses[ind] for ind in losses]

		# pass the new information to the optimizer
		self.last_result = self.tell(dim_values_of_losses, y=found_losses, fit=True)
		# self.all_results.append(self.last_result)

	def adapt_hyper_param(self, dim_values, verbose=True):
		adapt_hyper_param_name = self.adapt_param['param_name']
		min_adapt_param = int(self.hyper_param_dict[adapt_hyper_param_name]['min'])
		max_adapt_param = int(self.hyper_param_dict[adapt_hyper_param_name]['max'])
		par_cnt_scheme = self.adapt_param['par_cnt_scheme']
		param_thr = self.adapt_param['param_thr']
		vals_dict = {
			name: val for (name, val) in zip(self.hyper_param_names, dim_values)}

		# Exceptional case:
		if vals_dict['cnn_num_enc_lay'] == 0:
			# if no CNN part, output of LSTM should not be altered
			vals_dict['concat_flatten_last_2dims_cnn'] = 'True'
			dim_values[-1] = 'True'

		adapt_param_value = min_adapt_param
		prev_par_cnt_dict = dict()

		while True:
			vals_dict[adapt_hyper_param_name] = adapt_param_value

			values_suitable, par_cnt_dict = check_parameter_count(vals_dict, param_thr, par_cnt_scheme)

			if not values_suitable:
				if par_cnt_dict is None:
					best_adapt_param_value = adapt_param_value - 1
					break
				elif par_cnt_dict['total'] > param_thr:
					# went over allowed parameter count, best value for adaptation parameter is previous value
					best_adapt_param_value = adapt_param_value - 1
					break
			if values_suitable and vals_dict['cnn_num_enc_lay'] == 0:
				# there is no cnn, so no point in tuning the number of cnn filters. Just stop the adaptation and set
				# adaptation parameter to min_adapt_param
				best_adapt_param_value = min_adapt_param
				prev_par_cnt_dict = par_cnt_dict
				break
			if adapt_param_value > max_adapt_param:
				# reached maximum value vor adaptation parameter and still did not go over allowed_parameter_count*0.95
				best_adapt_param_value = max_adapt_param + 1
				break
			adapt_param_value += 1
			prev_par_cnt_dict = par_cnt_dict

		actual_par_cnt_dict = prev_par_cnt_dict

		if best_adapt_param_value < min_adapt_param or best_adapt_param_value > max_adapt_param or \
				actual_par_cnt_dict['total'] < param_thr*0.95:
			fixed_values_suitable = False
		else:
			fixed_values_suitable = True
			if verbose:
				print_str = 'Found suitable hyper parameter values, leading to %d number of trainable parameters (' % \
							actual_par_cnt_dict['total']
				for par_type, par_type_cnt in actual_par_cnt_dict.iteritems():
					if par_type != 'total':
						print_str += '%s: %d; ' % (par_type, par_type_cnt)
				print_str += ')'
				print print_str

		vals_dict[adapt_hyper_param_name] = best_adapt_param_value

		dim_values = [vals_dict[name] for name in self.hyper_param_names]

		return dim_values, fixed_values_suitable

	def prepare_configs(self, config_dir, requested_values):
		par_ind = 0
		alternative_format_lines = []
		for param_name in self.hyper_param_names:
			param_info = self.hyper_param_dict[param_name]

			param_info_keys = param_info.keys()
			standard_format = 'config' in param_info_keys and 'field' in param_info_keys and 'name' in param_info_keys
			alternative_format = any(['case' in key for key in param_info_keys])
			if not standard_format and not alternative_format:
				raise ValueError('No adjustments to the config files have been made for this parameter: %s' % param_name)

			if standard_format:
				config_file = os.path.join(config_dir, param_info['config'])
				param_config_field = param_info['field']
				param_config_name = param_info['name']
				config_data = configparser.ConfigParser()
				config_data.read(config_file)
				# check if multiple fields have to be set
				if ' ' in param_config_field:
					param_config_fields = param_config_field.split(' ')
					param_config_names = param_config_name.split(' ')
					if len(param_config_fields) != len(param_config_names):
						raise ValueError(
							'A config name should be set for each config field. Got %d config fields and %d config names' %
							(len(param_config_fields), len(param_config_names)))
					for (param_config_field, param_config_name) in zip(param_config_fields, param_config_names):
						config_data.set(param_config_field, param_config_name, str(requested_values[par_ind]))
				else:
					config_data.set(param_config_field, param_config_name, str(requested_values[par_ind]))

				with open(config_file, 'w') as fid:
					config_data.write(fid)

			if alternative_format:
				look_for_key = 'case_%s' % str(requested_values[par_ind]).lower()
				if look_for_key in param_info_keys:
					all_line_names = self.hyper_param_dict[param_name][look_for_key].split(' ')
					for line_name in all_line_names:
						alternative_format_lines.append(self.hyper_param_dict[param_name][line_name])

			par_ind += 1

		# Finally, apply the alternative format lines
		for line in alternative_format_lines:
			line_split = line.split(' ')
			config_file = line_split[0]
			config_file = os.path.join(config_dir, config_file)
			param_config_field = line_split[1]
			param_config_name = line_split[2]
			param_values = ' '.join(line_split[3:])
			config_data = configparser.ConfigParser()
			config_data.read(config_file)
			config_data.set(param_config_field, param_config_name, param_values)

			with open(config_file, 'w') as fid:
				config_data.write(fid)

	def checkpoint(self, optimizer_name='optimizer'):
		checkpoint_file = os.path.join(self.expdir, '%s.pkl' % optimizer_name)
		with open(checkpoint_file, 'w') as fid:
			pickle.dump(self, fid)

	def tell(self, x, y, fit=True):

		if fit:
			self.models = []
			# for ind in range(len(self.all_results)):
			# 	if self.all_results[ind]:
			# 		self.all_results[ind].models = []
		result = super(HyperParamOptimizer, self).tell(x, y, fit=fit)

		return result

	def _tell(self, x, y, fit=True):
		# Copied from skopt
		"""Perform the actual work of incorporating one or more new points.
		See `tell()` for the full description.

		This method exists to give access to the internals of adding points
		by side stepping all input validation and transformation."""

		if "ps" in self.acq_func:
			if is_2Dlistlike(x):
				self.Xi.extend(x)
				self.yi.extend(y)
				self._n_initial_points -= len(y)
			elif is_listlike(x):
				self.Xi.append(x)
				self.yi.append(y)
				self._n_initial_points -= 1
		# if y isn't a scalar it means we have been handed a batch of points
		elif is_listlike(y) and is_2Dlistlike(x):
			self.Xi.extend(x)
			self.yi.extend(y)
			self._n_initial_points -= len(y)
		elif is_listlike(x):
			self.Xi.append(x)
			self.yi.append(y)
			self._n_initial_points -= 1
		else:
			raise ValueError("Type of arguments `x` (%s) and `y` (%s) not compatible." % (type(x), type(y)))

		# optimizer learned something new - discard cache
		self.cache_ = {}

		# after being "told" n_initial_points we switch from sampling
		# random points to using a surrogate model
		if fit and self._n_initial_points <= 0 and self.base_estimator_ is not None:
			transformed_bounds = np.array(self.space.transformed_bounds)
			est = clone(self.base_estimator_)

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				est.fit(self.space.transform(self.Xi), self.yi)

			if hasattr(self, "next_xs_") and self.acq_func == "gp_hedge":
				self.gains_ -= est.predict(np.vstack(self.next_xs_))
			self.models.append(est)

			# We're gonna lie to the estimator by telling it a loss for the points that are still being evaluated,
			# similar to what we do when we ask for multiple points in ask().
			points_running = self.watch_list.keys()
			num_points_running = len(points_running)
			points_to_lie_about = [self.all_dim_values[run_ind] for run_ind in points_running]
			strategy = "cl_mean"
			if strategy == "cl_min":
				y_lie = np.min(self.yi) if self.yi else 0.0  # CL-min lie
			elif strategy == "cl_mean":
				y_lie = np.mean(self.yi) if self.yi else 0.0  # CL-mean lie
			else:
				y_lie = np.max(self.yi) if self.yi else 0.0  # CL-max lie
			# Lie to the fake optimizer.
			fake_est = copy.deepcopy(est)
			X_to_tell = self.Xi + points_to_lie_about
			X_to_tell = self.space.transform(X_to_tell)
			y_to_tell = self.yi + list(np.ones(num_points_running) * y_lie)

			fake_est.fit(X_to_tell, y_to_tell)

			# even with BFGS as optimizer we want to sample a large number
			# of points and then pick the best ones as starting points
			# X = self.space.transform(self.space.rvs(
			# 	n_samples=self.n_points, random_state=self.rng))
			Xspace = self.space.rvs(n_samples=self.n_points, random_state=self.rng)
			param_thr = self.adapt_param['param_thr']
			par_cnt_scheme = self.adapt_param['par_cnt_scheme']
			suitable_X, _ = check_parameter_count_for_sample(
					Xspace, self.hyper_param_names, param_thr, par_cnt_scheme)
			# for x in Xspace:
			# 	vals_suitable, _ = check_parameter_count_for_sample(
			# 		x, self.hyper_param_names, param_thr, par_cnt_scheme)
			# 	suitable_X.append(vals_suitable)

			Xspace = [Xspace[ind] for ind, suit in enumerate(suitable_X) if suit]
			X = self.space.transform(Xspace)

			self.next_xs_ = []
			for cand_acq_func in self.cand_acq_funcs_:
				values = _gaussian_acquisition(
					X=X, model=fake_est, y_opt=np.min(self.yi),
					acq_func=cand_acq_func,
					acq_func_kwargs=self.acq_func_kwargs)
				# Find the minimum of the acquisition function by randomly
				# sampling points from the space
				if self.acq_optimizer == "sampling":
					next_x = X[np.argmin(values)]

				# Use BFGS to find the mimimum of the acquisition function, the
				# minimization starts from `n_restarts_optimizer` different
				# points and the best minimum is used
				elif self.acq_optimizer == "lbfgs":
					x0 = X[np.argsort(values)[:self.n_restarts_optimizer]]

					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						results = Parallel(n_jobs=self.n_jobs)(
							delayed(fmin_l_bfgs_b)(
								gaussian_acquisition_1D, x,
								args=(fake_est, np.min(self.yi), cand_acq_func, self.acq_func_kwargs),
								bounds=self.space.transformed_bounds,
								approx_grad=False,
								maxiter=20)
							for x in x0)

					cand_xs = np.array([r[0] for r in results])
					cand_acqs = np.array([r[1] for r in results])
					next_x = cand_xs[np.argmin(cand_acqs)]

				# lbfgs should handle this but just in case there are
				# precision errors.
				if not self.space.is_categorical:
					next_x = np.clip(
						next_x, transformed_bounds[:, 0],
						transformed_bounds[:, 1])
				self.next_xs_.append(next_x)

			if self.acq_func == "gp_hedge":
				logits = np.array(self.gains_)
				logits -= np.max(logits)
				exp_logits = np.exp(self.eta * logits)
				probs = exp_logits / np.sum(exp_logits)
				next_x = self.next_xs_[np.argmax(self.rng.multinomial(1, probs))]
			else:
				next_x = self.next_xs_[0]

			# note the need for [0] at the end
			self._next_x = self.space.inverse_transform(
				next_x.reshape((1, -1)))[0]

		# Pack results
		return create_result(self.Xi, self.yi, self.space, self.rng, models=self.models)

	def copy(self, random_state=None):
		"""Create a shallow copy of an instance of the optimizer.

		Parameters
		----------
		* `random_state` [int, RandomState instance, or None (default)]:
			Set the random state of the copy.
		"""

		optimizer = HyperParamOptimizer(
			hyper_param_conf=self.hyper_param_conf, command=self.command, expdir=self.expdir,
			exp_recipe_dir=self.exp_recipe_dir, recipe=self.recipe, computing=self.computing,
			exp_proposal_watch_dir=self.exp_proposal_watch_dir)

		super(HyperParamOptimizer, optimizer).__init__(
			dimensions=self.space.dimensions,
			base_estimator=self.base_estimator_,
			n_initial_points=self.n_initial_points_,
			acq_func=self.acq_func,
			acq_optimizer=self.acq_optimizer,
			acq_func_kwargs=self.acq_func_kwargs,
			acq_optimizer_kwargs=self.acq_optimizer_kwargs,
			random_state=random_state)
		optimizer.n_points = self.n_points

		if hasattr(self, "gains_"):
			optimizer.gains_ = np.copy(self.gains_)

		if self.Xi:
			optimizer._tell(self.Xi, self.yi)

		return optimizer

	def create_opt_only_val_loss(self):
		new_opt = copy.deepcopy(self)
		new_opt.Xi = []
		new_opt.yi = []
		new_opt.n_points = 4000
		new_opt.process_losses(new_opt.all_losses)
		new_opt.checkpoint('optimizer_only_valid_losses')
