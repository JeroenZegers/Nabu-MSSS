import warnings
import numpy as np
import copy

from skopt.space import  space as skopt_space
from skopt.learning import GaussianProcessRegressor
from scipy.linalg import cho_solve
from sklearn.utils.validation import check_array

from utils import check_parameter_count


class BoundedGaussianProcessRegressor(GaussianProcessRegressor):
	"""
	Gaussian process regressor where part of the space in not allowed due to parameter count.
	"""
	def __init__(self, space, hyper_param_names, adapt_param, kernel, normalize_y, noise, n_restarts_optimizer):
		self.space = space
		self.hyper_param_names = hyper_param_names
		self.adapt_param = adapt_param
		self.param_thr = adapt_param['param_thr']
		self.par_cnt_scheme = adapt_param['par_cnt_scheme']

		transformed_categorical = []
		for dim in space.dimensions:
			if not isinstance(dim, skopt_space.Categorical):
				transformed_categorical.append(False)
			else:
				transformed_categorical += [True] * dim.transformed_size
		if len(transformed_categorical) != space.transformed_n_dims:
			raise Exception()

		self.transformed_categorical = transformed_categorical

		super(BoundedGaussianProcessRegressor, self).__init__(
			kernel=kernel, normalize_y=normalize_y, noise=noise, n_restarts_optimizer=n_restarts_optimizer)

	def predict(self, X, return_std=False, return_cov=False, return_mean_grad=False, return_std_grad=False):
		"""
		Predict output for X.
		In addition to the mean of the predictive distribution, also its
		standard deviation (return_std=True) or covariance (return_cov=True),
		the gradient of the mean and the standard-deviation with respect to X
		can be optionally provided.
		Parameters
		----------
		* `X` [array-like, shape = (n_samples, n_features)]:
			Query points where the GP is evaluated.
		* `return_std` [bool, default: False]:
			If True, the standard-deviation of the predictive distribution at
			the query points is returned along with the mean.
		* `return_cov` [bool, default: False]:
			If True, the covariance of the joint predictive distribution at
			the query points is returned along with the mean.
		* `return_mean_grad` [bool, default: False]:
			Whether or not to return the gradient of the mean.
			Only valid when X is a single point.
		* `return_std_grad` [bool, default: False]:
			Whether or not to return the gradient of the std.
			Only valid when X is a single point.
		Returns
		-------
		* `y_mean` [array, shape = (n_samples, [n_output_dims]):
			Mean of predictive distribution a query points
		* `y_std` [array, shape = (n_samples,), optional]:
			Standard deviation of predictive distribution at query points.
			Only returned when return_std is True.
		* `y_cov` [array, shape = (n_samples, n_samples), optional]:
			Covariance of joint predictive distribution a query points.
			Only returned when return_cov is True.
		* `y_mean_grad` [shape = (n_samples, n_features)]:
			The gradient of the predicted mean
		* `y_std_grad` [shape = (n_samples, n_features)]:
			The gradient of the predicted std.
		"""
		if return_std and return_cov:
			raise RuntimeError(
				"Not returning standard deviation of predictions when "
				"returning full covariance.")

		if return_std_grad and not return_std:
			raise ValueError(
				"Not returning std_gradient without returning "
				"the std.")

		X = check_array(X)
		if X.shape[0] != 1 and (return_mean_grad or return_std_grad):
			raise ValueError("Not implemented for n_samples > 1")

		# check if X is within bounds defined by parameter count
		for ind1, x in enumerate(X):
			for ind2, xi in enumerate(x):
				if xi < 0 or xi > 1:
					if -1e-10 < xi < 0:
						X[ind1][ind2] = 0
					elif 1 < xi < 1 + 1e-10:
						X[ind1][ind2] = 1
					else:
						raise Exception('Not al points in space')

		all_dim_values = self.space.inverse_transform(X)
		all_vals_dict = [
			{name: val for (name, val) in zip(self.hyper_param_names, dim_values)} for dim_values in all_dim_values]

		suitable_inds = []
		unsuitable_inds = []
		all_par_cnt_dict = []
		for ind, vals_dict in enumerate(all_vals_dict):
			values_suitable, par_cnt_dict = check_parameter_count(vals_dict, self.param_thr, self.par_cnt_scheme)
			all_par_cnt_dict.append(par_cnt_dict)
			if values_suitable:
				suitable_inds.append(ind)
			else:
				unsuitable_inds.append(ind)
		if not all_par_cnt_dict:
			raise ValueError()

		X_suit = np.array([X[ind] for ind in suitable_inds])
		X_unsuit = np.array([X[ind] for ind in unsuitable_inds])

		if len(X_suit) > 0:
			if not hasattr(self, "X_train_"):  # Not fit; predict based on GP prior
				y_suit_mean = np.zeros(X_suit.shape[0])
				if return_cov:
					y_suit_cov = self.kernel(X_suit)
				elif return_std:
					y_suit_var = self.kernel.diag(X_suit)
					y_suit_std = np.sqrt(y_suit_var)

			else:  # Predict based on GP posterior
				K_trans = self.kernel_(X_suit, self.X_train_)
				y_suit_mean = K_trans.dot(self.alpha_)    # Line 4 (y_suit_mean = f_star)
				y_suit_mean = self.y_train_mean_ + y_suit_mean  # undo normal.

				if return_cov:
					v = cho_solve((self.L_, True), K_trans.T)  # Line 5
					y_suit_cov = self.kernel_(X_suit) - K_trans.dot(v)   # Line 6

				elif return_std:
					K_inv = self.K_inv_

					# Compute variance of predictive distribution
					y_suit_var = self.kernel_.diag(X_suit)
					y_suit_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv)

					# Check if any of the variances is negative because of
					# numerical issues. If yes: set the variance to 0.
					y_suit_var_negative = y_suit_var < 0
					if np.any(y_suit_var_negative):
						warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
						y_suit_var[y_suit_var_negative] = 0.0
					y_suit_std = np.sqrt(y_suit_var)

				if return_mean_grad:
					grad_suit = self.kernel_.gradient_x(X_suit[0], self.X_train_)
					grad_suit_mean = np.dot(grad_suit.T, self.alpha_)

					if return_std_grad:
						grad_suit_std = np.zeros(X_suit.shape[1])
						if not np.allclose(y_suit_std, grad_suit_std):
							grad_suit_std = -np.dot(K_trans, np.dot(K_inv, grad_suit))[0] / y_suit_std

		else:
			y_suit_mean = []
			y_suit_cov = []
			y_suit_std = []
			grad_suit_mean = []
			grad_suit_std = []

		eps = 1e-10
		y_unsuit_mean = []
		grad_unsuit_mean = []
		y_unsuit_std = [eps] * len(X_unsuit)
		y_unsuit_cov = [eps] * len(X_unsuit)
		grad_unsuit_std = list(np.ones(np.shape(X_unsuit)) * eps)
		y_max = np.max(self.y_train_)
		for unsuit_ind, x in zip(unsuitable_inds, X_unsuit):
			try:
				tot_par = all_par_cnt_dict[unsuit_ind]['total']
			except:
				print 1
			tmp_y_unsuit_mean = y_overshoot(tot_par, self.param_thr, y_max)
			y_unsuit_mean.append(tmp_y_unsuit_mean)

			if return_mean_grad:
				tmp_grad_unsuit_mean = []
				for dim_ind, is_categorical in enumerate(self.transformed_categorical):
					if is_categorical:
						tmp_grad_unsuit_mean.append(eps)
					else:
						# find a numerical gradient: grad_i = [y(xi+delta_xi) - y(xi)] / delta_xi
						xi = x[dim_ind]
						delta_xi = 0.2
						x2i = xi + delta_xi
						if x2i > 1:
							x2i = 1.0
							delta_xi = x2i - xi
							if delta_xi <= 0:
								tmp_grad_unsuit_mean.append(eps)
								continue
						x2 = copy.deepcopy(x)
						x2[dim_ind] = x2i
						dim_values2 = self.space.inverse_transform(np.array([x2]))[0]
						vals_dict2 = {name: val for (name, val) in zip(self.hyper_param_names, dim_values2)}
						values_suitable2, par_cnt_dict2 = check_parameter_count(
							vals_dict2, self.param_thr, self.par_cnt_scheme)
						tot_par2 = par_cnt_dict2['total']
						if values_suitable2:
							tot_par2 = self.param_thr + 1
						y2 = y_overshoot(tot_par2, self.param_thr, y_max)

						grad_unsuit_i = (y2 - tmp_y_unsuit_mean)/delta_xi
						tmp_grad_unsuit_mean.append(grad_unsuit_i)
				grad_unsuit_mean.append(tmp_grad_unsuit_mean)

		y_mean = []
		y_std = []
		y_cov = []
		grad_mean = []
		grad_std = []
		for ind in range(len(X)):
			if ind in suitable_inds:
				pos = [i for i, x in enumerate(suitable_inds) if x == ind]
				pos = pos[0]
				y_mean.append(y_suit_mean[pos])
				if return_cov:
					y_cov.append(y_suit_cov[pos])
				if return_std:
					y_std.append(y_suit_std[pos])
				if return_mean_grad:
					# only one value allowed
					grad_mean.append(grad_suit_mean)
				if return_std_grad:
					# only one value allowed
					grad_std.append(grad_suit_std)

			elif ind in unsuitable_inds:
				pos = [i for i, x in enumerate(unsuitable_inds) if x == ind]
				pos = pos[0]
				y_mean.append(y_unsuit_mean[pos])
				if return_cov:
					y_cov.append(y_unsuit_cov[pos])
				if return_std:
					y_std.append(y_unsuit_std[pos])
				if return_mean_grad:
					grad_mean.append(grad_unsuit_mean[pos])
				if return_std_grad:
					grad_std.append(grad_unsuit_std[pos])
			else:
				raise Exception()

		y_mean = np.array(y_mean)
		y_std = np.array(y_std)
		y_cov = np.array(y_cov)
		if return_mean_grad:
			grad_mean = np.array(grad_mean[0])
		if return_std_grad:
			grad_std = np.array(grad_std[0])

		if return_cov:
			return y_mean, y_cov
		if return_mean_grad:
			if return_std_grad:
				return y_mean, y_std, grad_mean, grad_std
			if return_std:
				return y_mean, y_std, grad_mean
			else:
				return y_mean, grad_mean
		else:
			if return_std:
				return y_mean, y_std
			else:
				return y_mean


def y_overshoot(tot_par, param_thr, y_max):
	eps = 1e-10
	y_max = max([y_max, eps])
	if tot_par > param_thr:
		par_overshoot = (tot_par - param_thr)/param_thr
	elif tot_par < param_thr * 0.95:
		par_overshoot = -(tot_par - param_thr * 0.95)/param_thr * 0.95
	else:
		raise Exception()

	y = y_max * 1.1 + max([y_max, 10*eps]) * 0.1 * par_overshoot
	# y = y_max * (1.1 + 0.1 * par_overshoot)

	return y
