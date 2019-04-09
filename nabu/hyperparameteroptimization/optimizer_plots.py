import numpy as np
import matplotlib
from matplotlib import pyplot as plt, tri as tri
from matplotlib.ticker import LogLocator, MaxNLocator
from skopt import plots as skopt_plots

import plotly.plotly as plotlypy
import plotly.tools as tls

from utils import check_parameter_count_for_sample, partial_dependence_valid_samples_allow_paramcounts


def plot_objective(
		result, param_thr, hyper_param_names,  par_cnt_scheme='enc_dec_cnn_lstm_ff', levels=10, n_points=20,
		n_samples=250, size=2, zscale='linear', selected_dimensions=None, dimensions=None):
	"""
	Copied from skopt and altered
	Pairwise partial dependence plot of the objective function.

	The diagonal shows the partial dependence for dimension `i` with
	respect to the objective function. The off-diagonal shows the
	partial dependence for dimensions `i` and `j` with
	respect to the objective function. The objective function is
	approximated by `result.model.`

	Pairwise scatter plots of the points at which the objective
	function was directly evaluated are shown on the off-diagonal.
	A red point indicates the found minimum.

	Note: search spaces that contain `Categorical` dimensions are currently not supported by this function.

	Parameters
	----------
	* `result` [`OptimizeResult`]
		The result for which to create the scatter plot matrix.

	* `param_thr` [int]
		threshold on trainable parameter count

	* `hyper_param_names`
		Names of the hyper parameters

	* `par_cnt_scheme` [default='enc_dec_cnn_lstm_ff']
		The scheme to use to count the trainable parameters, to check if a sample is valid.

	* `levels` [int, default=10]
		Number of levels to draw on the contour plot, passed directly
		to `plt.contour()`.

	* `n_points` [int, default=20]
		Number of points at which to evaluate the partial dependence
		along each dimension.

	* `n_samples` [int, default=250]
		Number of random samples to use for averaging the model function
		at each of the `n_points`.

	* `size` [float, default=2]
		Height (in inches) of each facet.

	* `zscale` [str, default='linear']
		Scale to use for the z axis of the contour plots. Either 'linear'
		or 'log'..

	* `selected_dimensions` [list, default='None']
		Dimensions chosen to plot. If 'None', plot all dimensions

	* `dimensions` [list of str, default=None] Labels of the dimension
		variables. `None` defaults to `space.dimensions[i].name`, or
		if also `None` to `['X_0', 'X_1', ..]`.

	Returns
	-------
	* `ax`: [`Axes`]:
		The matplotlib axes.
	"""
	space = result.space
	exps = result.x_iters
	_, exps_par_dicts = check_parameter_count_for_sample(exps, hyper_param_names, param_thr, par_cnt_scheme)
	_, res_x_par_dict = check_parameter_count_for_sample(result.x, hyper_param_names, param_thr, par_cnt_scheme)
	exps = np.asarray(exps)
	# samples = space.rvs(n_samples=n_samples)
	# suitable_samples, _ = check_parameter_count_for_sample(samples, hyper_param_names, param_thr, par_cnt_scheme)
	#
	# samples = [samples[ind] for ind, suit in enumerate(suitable_samples) if suit]
	#
	# rvs_transformed = space.transform(samples)

	if selected_dimensions is None:
		selected_dimensions = range(space.n_dims)
	n_dims = len(selected_dimensions)

	if dimensions is not None:
		dimensions = [dimensions[ind] for ind in selected_dimensions]

	if zscale == 'log':
		locator = LogLocator()
	elif zscale == 'linear':
		locator = None
	else:
		raise ValueError("Valid values for zscale are 'linear' and 'log', not '%s'." % zscale)

	fig, ax = plt.subplots(n_dims, n_dims, figsize=(size * n_dims, size * n_dims))

	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)

	for i, dim_i in enumerate(selected_dimensions):
		for j, dim_j in enumerate(selected_dimensions):
			if i == j:
				xi, yi, yi_std = partial_dependence_valid_samples_allow_paramcounts(
					space, result.models[-1], param_thr, hyper_param_names, dim_i, j=None,
					par_cnt_scheme=par_cnt_scheme, n_samples=n_samples, n_points=n_points)

				ax[i, i].plot(xi, yi)
				ax[i, i].plot(xi, yi+yi_std, ':', color='C0')
				ax[i, i].plot(xi, yi-yi_std, ':', color='C0')
				if isinstance(dim_i, str):
					# using parameter counts.
					res_xi = res_x_par_dict[dim_i]
				else:
					res_xi = result.x[dim_i]
				ax[i, i].axvline(res_xi, linestyle="--", color="r", lw=1)

			# lower triangle
			elif i > j:
				xi, yi, zi = partial_dependence_valid_samples_allow_paramcounts(
					space, result.models[-1], param_thr, hyper_param_names, dim_i, dim_j, par_cnt_scheme,
					n_samples=n_samples, n_points=n_points)
				cont = ax[i, j].contourf(xi, yi, zi, levels, locator=locator, cmap='viridis_r')
				plt.colorbar(cont, ax=ax[i, j])
				if isinstance(dim_j, str):
					# using parameter counts.
					x_param_cnt = True
					exps_xi = [exp_par_dict[dim_j] for exp_par_dict in exps_par_dicts]
					res_xi = res_x_par_dict[dim_j]
				else:
					x_param_cnt = False
					exps_xi = exps[:, dim_j]
					res_xi = result.x[dim_j]
				if isinstance(dim_i, str):
					# using parameter counts.
					y_param_cnt = True
					exps_yi = [exp_par_dict[dim_i] for exp_par_dict in exps_par_dicts]
					res_yi = res_x_par_dict[dim_i]
				else:
					y_param_cnt = False
					exps_yi = exps[:, dim_i]
					res_yi = result.x[dim_i]

				ax[i, j].scatter(exps_xi, exps_yi, c='k', s=10, lw=0.)
				ax[i, j].scatter(res_xi, res_yi, c=['r'], s=20, lw=0.)

	complete_ax = _format_scatter_plot_axes(
		ax, space, ylabel="Partial dependence", param_thr=param_thr, selected_dimensions=selected_dimensions,
		dim_labels=dimensions)

	fig_name = 'figures/dim'
	for dim in selected_dimensions:
		fig_name = '%s_%s' % (fig_name, str(dim))
	fig_name = '%s_%d.png' % (fig_name, n_samples)

	fig.set_size_inches((16, 11), forward=False)
	plt.savefig(fig_name, dpi=700)

	return complete_ax


def _format_scatter_plot_axes(ax, space, ylabel, param_thr, selected_dimensions=None, dim_labels=None):
	if selected_dimensions is None:
		selected_dimensions = range(space.n_dims)
	n_dims = len(selected_dimensions)

	# Work out min, max of y axis for the diagonal so we can adjust
	# them all to the same value
	diagonal_ylim = (
		np.min([ax[i, i].get_ylim()[0] for i in range(n_dims)]),
		np.max([ax[i, i].get_ylim()[1] for i in range(n_dims)]))

	if dim_labels is None:
		dim_labels = ["$X_{%i}$" % i if d.name is None else d.name for i, d in enumerate(space.dimensions)]

	# Deal with formatting of the axes
	for i, dim_i in enumerate(selected_dimensions):
		for j, dim_j in enumerate(selected_dimensions):
			ax_ = ax[i, j]

			if j > i:
				ax_.axis("off")

			# off-diagonal axis
			if i != j:
				if isinstance(dim_j, str):
					# using parameter counts.
					x_param_cnt = True
				else:
					x_param_cnt = False
				if isinstance(dim_i, str):
					# using parameter counts.
					y_param_cnt = True
				else:
					y_param_cnt = False
				# plots on the diagonal are special, like Texas. They have
				# their own range so do not mess with them.
				if y_param_cnt:
					ax_.set_ylim([0, param_thr])
				else:
					ax_.set_ylim(*space.dimensions[dim_i].bounds)
				if x_param_cnt:
					ax_.set_xlim([0, param_thr])
				else:
					ax_.set_xlim(*space.dimensions[dim_j].bounds)

				if j > 0:
					ax_.set_yticklabels([])
				else:
					ax_.set_ylabel(dim_labels[i])

				# for all rows except ...
				if i < n_dims - 1:
					ax_.set_xticklabels([])
				# ... the bottom row
				else:
					[l.set_rotation(45) for l in ax_.get_xticklabels()]
					ax_.set_xlabel(dim_labels[j])

				# configure plot for linear vs log-scale
				if y_param_cnt:
					y_prior = ''
				else:
					y_prior = space.dimensions[dim_i].prior
				if x_param_cnt:
					x_prior = ''
				else:
					x_prior = space.dimensions[dim_j].prior

				priors = (x_prior, y_prior)
				scale_setters = (ax_.set_xscale, ax_.set_yscale)
				loc_setters = (ax_.xaxis.set_major_locator, ax_.yaxis.set_major_locator)
				for set_major_locator, set_scale, prior in zip(
						loc_setters, scale_setters, priors):
					if prior == 'log-uniform':
						set_scale('log')
					else:
						set_major_locator(MaxNLocator(6, prune='both'))

			else:
				ax_.set_ylim(*diagonal_ylim)
				ax_.yaxis.tick_right()
				ax_.yaxis.set_label_position('right')
				ax_.yaxis.set_ticks_position('both')
				ax_.set_ylabel(ylabel)

				ax_.xaxis.tick_top()
				ax_.xaxis.set_label_position('top')
				ax_.set_xlabel(dim_labels[j])

				if space.dimensions[i].prior == 'log-uniform':
					ax_.set_xscale('log')
				else:
					ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both'))

	return ax


def plot_flat_objective(
		result, param_thr, hyper_param_names,  par_cnt_scheme='enc_dec_cnn_lstm_ff', levels=10, n_points=20,
		n_samples=250, size=2, zscale='linear', selected_dimensions=None, dimensions=None):
	"""
	Copied from skopt and altered
	Pairwise partial dependence plot of the objective function.

	The diagonal shows the partial dependence for dimension `i` with
	respect to the objective function. The off-diagonal shows the
	partial dependence for dimensions `i` and `j` with
	respect to the objective function. The objective function is
	approximated by `result.model.`

	Pairwise scatter plots of the points at which the objective
	function was directly evaluated are shown on the off-diagonal.
	A red point indicates the found minimum.

	Note: search spaces that contain `Categorical` dimensions are currently not supported by this function.

	Parameters
	----------
	* `result` [`OptimizeResult`]
		The result for which to create the scatter plot matrix.

	* `param_thr` [int]
		threshold on trainable parameter count

	* `hyper_param_names`
		Names of the hyper parameters

	* `par_cnt_scheme` [default='enc_dec_cnn_lstm_ff']
		The scheme to use to count the trainable parameters, to check if a sample is valid.

	* `levels` [int, default=10]
		Number of levels to draw on the contour plot, passed directly
		to `plt.contour()`.

	* `n_points` [int, default=20]
		Number of points at which to evaluate the partial dependence
		along each dimension.

	* `n_samples` [int, default=250]
		Number of random samples to use for averaging the model function
		at each of the `n_points`.

	* `size` [float, default=2]
		Height (in inches) of each facet.

	* `zscale` [str, default='linear']
		Scale to use for the z axis of the contour plots. Either 'linear'
		or 'log'..

	* `selected_dimensions` [list, default='None']
		Dimensions chosen to plot. If 'None', plot all dimensions

	* `dimensions` [list of str, default=None] Labels of the dimension
		variables. `None` defaults to `space.dimensions[i].name`, or
		if also `None` to `['X_0', 'X_1', ..]`.

	Returns
	-------
	* `ax`: [`Axes`]:
		The matplotlib axes.
	"""
	space = result.space
	exps = np.asarray(result.x_iters)
	_, exps_par_dicts = check_parameter_count_for_sample(exps, hyper_param_names, param_thr, par_cnt_scheme)
	_, res_x_par_dict = check_parameter_count_for_sample(result.x, hyper_param_names, param_thr, par_cnt_scheme)
	# samples = space.rvs(n_samples=n_samples)
	# suitable_samples, _ = check_parameter_count_for_sample(samples, hyper_param_names, param_thr, par_cnt_scheme)
	#
	# samples = [samples[ind] for ind, suit in enumerate(suitable_samples) if suit]
	#
	# rvs_transformed = space.transform(samples)

	if selected_dimensions is None:
		selected_dimensions = range(space.n_dims)
	n_dims = len(selected_dimensions)

	if dimensions is not None:
		dimensions = [dimensions[ind] for ind in selected_dimensions]

	if zscale == 'log':
		locator = LogLocator()
	elif zscale == 'linear':
		locator = None
	else:
		raise ValueError("Valid values for zscale are 'linear' and 'log', not '%s'." % zscale)

	fig, ax = plt.subplots(n_dims, n_dims, figsize=(size * n_dims, size * n_dims))

	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)

	for i, dim_i in enumerate(selected_dimensions):
		for j, dim_j in enumerate(selected_dimensions):
			if i == j:
				xi, yi, yi_std = partial_dependence_valid_samples_allow_paramcounts(
					space, result.models[-1], param_thr, hyper_param_names, dim_i, j=None,
					par_cnt_scheme=par_cnt_scheme, n_samples=n_samples, n_points=n_points)
				# xi = exps[:, dim_i]
				# yi = result.func_vals
				#
				# sort_inds = np.argsort(xi)
				# xi_sorted = [xi[ind] for ind in sort_inds]
				# yi_sorted = [yi[ind] for ind in sort_inds]

				ax[i, i].plot(xi, yi)
				ax[i, i].plot(xi, yi+yi_std, ':', color='C0')
				ax[i, i].plot(xi, yi-yi_std, ':', color='C0')
				ax[i, i].axvline(result.x[dim_i], linestyle="--", color="r", lw=1)

			# lower triangle
			elif i > j:
				# xi, yi, zi = partial_dependence_valid_samples_allow_paramcounts(
				# 	space, result.models[-1], param_thr, hyper_param_names, dim_i, dim_j, par_cnt_scheme,
				# 	rvs_transformed, n_points)

				x = exps[:, dim_j]
				x = [float(xi) for xi in x]
				y = exps[:, dim_i]
				y = [float(yi) for yi in y]
				z = result.func_vals

				x_range = max(x) - min(x)
				y_range = max(y) - min(y)
				xi = np.linspace(min(x)-x_range/10, max(x)+x_range/10, 1000)
				yi = np.linspace(min(y)-y_range/10, max(y)+y_range/10, 1000)

				# Perform linear interpolation of the data (x,y)
				# on a grid defined by (xi,yi)
				triang = tri.Triangulation(x, y)
				interpolator = tri.LinearTriInterpolator(triang, z)
				Xi, Yi = np.meshgrid(xi, yi)
				zi = interpolator(Xi, Yi)

				# Note that scipy.interpolate provides means to interpolate data on a grid
				# as well. The following would be an alternative to the four lines above:
				# from scipy.interpolate import griddata
				# zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')

				# ax[i, j].contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
				cntr1 = ax[i, j].contourf(xi, yi, zi, levels, locator=locator, cmap="viridis_r")

				fig.colorbar(cntr1, ax=ax[i, j])

				# cont = ax[i, j].contourf(xi, yi, zi, levels, locator=locator, cmap='viridis_r')
				# plt.colorbar(cont, ax=ax[i, j])
				if isinstance(dim_j, str):
					# using parameter counts.
					x_param_cnt = True
					exps_xi = [exp_par_dict[dim_j] for exp_par_dict in exps_par_dicts]
					res_xi = res_x_par_dict[dim_j]
				else:
					x_param_cnt = False
					exps_xi = exps[:, dim_j]
					res_xi = result.x[dim_j]
				if isinstance(dim_i, str):
					# using parameter counts.
					y_param_cnt = True
					exps_yi = [exp_par_dict[dim_i] for exp_par_dict in exps_par_dicts]
					res_yi = res_x_par_dict[dim_i]
				else:
					y_param_cnt = False
					exps_yi = exps[:, dim_i]
					res_yi = result.x[dim_i]

				ax[i, j].scatter(exps_xi, exps_yi, c='k', s=10, lw=0.)
				ax[i, j].scatter(res_xi, res_yi, c=['r'], s=20, lw=0.)

	return _format_scatter_plot_axes(
		ax, space, ylabel="Partial dependence", selected_dimensions=selected_dimensions, dim_labels=dimensions)


def plot_param_count(optimizer, result):
	all_dim_values = result.x_iters
	losses = result.func_vals

	par_cnt_scheme = optimizer.adapt_param['par_cnt_scheme']
	param_thr = optimizer.adapt_param['param_thr']

	all_par_dicts = []

	bad_param_cnt_inds = []
	bad_param_dict = []
	for ind, dim_values in enumerate(all_dim_values):
		_, par_dict = check_parameter_count_for_sample(
			dim_values, optimizer.hyper_param_names, param_thr, par_cnt_scheme=par_cnt_scheme)
		all_par_dicts.append(par_dict)
		if par_dict['total'] < param_thr*0.95 or par_dict['total'] > param_thr:
			bad_param_cnt_inds.append(ind)
			bad_param_dict.append(par_dict)

	sorted_inds = np.argsort(losses)
	sorted_losses = [losses[ind] for ind in sorted_inds]
	sorted_par_dicts = [all_par_dicts[ind] for ind in sorted_inds]
	cnn_cnts = np.array([par_dict['cnn'] for par_dict in sorted_par_dicts])
	lstm_cnts = np.array([par_dict['lstm'] for par_dict in sorted_par_dicts])
	ff_cnts = np.array([par_dict['ff'] for par_dict in sorted_par_dicts])

	n_exps = len(sorted_losses)
	exps_range = np.arange(n_exps)

	font = {'size': 16}
	matplotlib.rc('font', **font)
	# matplotlib.rcParams.update({'font.size': 22})

	mpl_fig = plt.figure()
	ax_loss = mpl_fig.add_subplot(211)
	ax_loss.scatter(exps_range, sorted_losses)
	ax_loss.set_ylabel('Validation loss')
	# ax_loss.plot(sorted_losses)

	ax = mpl_fig.add_subplot(212, sharex=ax_loss)
	width = 0.50
	p1 = ax.bar(exps_range, cnn_cnts, width)
	p2 = ax.bar(exps_range, lstm_cnts, width, bottom=cnn_cnts)
	p3 = ax.bar(exps_range, ff_cnts, width, bottom=lstm_cnts+cnn_cnts)

	ax.legend(['CNN', 'LSTM', 'FC'], loc='upper right')
	ax.set_ylabel('Trainable parameter count (in millions)')
	ax.set_xlabel('Experiments ordered by loss')
	# ax.set_title('Trainable parameter count per experiment, ordered by loss')

	# ax.set_xticks(np.arange(0, n_exps+7, 20))
	# ax.set_xticks(exps_range)
	# # ax.set_xticks(exps_range + width/2.)
	# ax.set_xticklabels(exps_range)
	y_ticks = np.arange(0, param_thr, 2e6)
	y_tickslabels = np.arange(0, param_thr/1e6, 2e6/1e6)
	y_tickslabels = [str(lab) for lab in y_tickslabels]
	ax.set_yticks(y_ticks)
	ax.set_yticklabels(y_tickslabels)

	# make vertical line befor experiments with loss>0.17
	tmp = [ind for ind, loss in enumerate(sorted_losses) if loss >= 0.17]

	ax_loss.axvline(tmp[0]-0.5, linestyle="-", lw=1)
	ax.axvline(tmp[0]-0.5, linestyle="-", lw=1)

	# plotly_fig = tls.mpl_to_plotly(mpl_fig)
	#
	# # For Legend
	# plotly_fig["layout"]["showlegend"] = True
	# plotly_fig["data"][0]["name"] = "CNN param count"
	# plotly_fig["data"][1]["name"] = "LSTM param count"
	# plotly_fig["data"][2]["name"] = "Output param count"
	# plotlypy.iplot(plotly_fig, filename='stacked-bar-chart')
	# 1


def plot_param_count_and_window_size(optimizer, result):
	all_dim_values = result.x_iters
	losses = result.func_vals

	par_cnt_scheme = optimizer.adapt_param['par_cnt_scheme']
	param_thr = optimizer.adapt_param['param_thr']

	all_par_dicts = []

	bad_param_cnt_inds = []
	bad_param_dict = []
	for ind, dim_values in enumerate(all_dim_values):
		_, par_dict = check_parameter_count_for_sample(
			dim_values, optimizer.hyper_param_names, param_thr, par_cnt_scheme=par_cnt_scheme)
		all_par_dicts.append(par_dict)
		if par_dict['total'] < param_thr*0.95 or par_dict['total'] > param_thr:
			bad_param_cnt_inds.append(ind)
			bad_param_dict.append(par_dict)

	sorted_inds = np.argsort(losses)
	sorted_losses = [losses[ind] for ind in sorted_inds]
	sorted_par_dicts = [all_par_dicts[ind] for ind in sorted_inds]
	cnn_cnts = np.array([par_dict['cnn'] for par_dict in sorted_par_dicts])
	lstm_cnts = np.array([par_dict['lstm'] for par_dict in sorted_par_dicts])
	ff_cnts = np.array([par_dict['ff'] for par_dict in sorted_par_dicts])

	win_t_sizes = np.array([all_dim_values[ind][4] for ind in sorted_inds])
	win_f_sizes = np.array([all_dim_values[ind][6] for ind in sorted_inds])

	n_exps = len(sorted_losses)
	exps_range = np.arange(n_exps)

	mpl_fig = plt.figure()
	ax_loss = mpl_fig.add_subplot(311)
	ax_loss.scatter(exps_range, sorted_losses)
	# ax_loss.plot(sorted_losses)
	ymin = np.floor((np.min(sorted_losses)-1e-10)*100)/100
	ymax = np.ceil((np.max(sorted_losses)+1e-10)*100)/100
	ax_loss.set_ylim([ymin, ymax])
	ax_loss.set_ylabel('Validation loss')

	ax = mpl_fig.add_subplot(312)
	width = 0.50
	p1 = ax.bar(exps_range, cnn_cnts, width)
	p2 = ax.bar(exps_range, lstm_cnts, width, bottom=cnn_cnts)
	p3 = ax.bar(exps_range, ff_cnts, width, bottom=lstm_cnts+cnn_cnts)

	ax.set_ylabel('Trainable parameter count (in millions)')
	# ax.set_xlabel('Experiments ordered by loss')
	# ax.set_title('Trainable parameter count per experiment, ordered by loss')

	ax.set_xticks(exps_range)
	# ax.set_xticks(exps_range + width/2.)
	ax.set_xticklabels(exps_range)
	y_ticks = np.arange(0, param_thr, 2e6)
	y_tickslabels = np.arange(0, param_thr/1e6, 2e6/1e6)
	y_tickslabels = [str(lab) for lab in y_tickslabels]
	ax.set_yticks(y_ticks)
	ax.set_yticklabels(y_tickslabels)

	#

	ax_win = mpl_fig.add_subplot(313)
	width = 0.50
	p1 = ax_win.bar(exps_range, win_t_sizes, width)
	p2 = ax_win.bar(exps_range, win_f_sizes, width, bottom=win_t_sizes)

	ax_win.set_ylabel('Convolutional window sizes')
	ax_win.set_xlabel('Experiments ordered by loss')
	# ax_win.set_title('Convolutional window sizes')

	ax_win.set_xticks(exps_range)
	# ax.set_xticks(exps_range + width/2.)
	ax_win.set_xticklabels(exps_range)
	# y_ticks = np.arange(0, param_thr, 2e6)
	# y_tickslabels = np.arange(0, param_thr / 1e6, 2e6 / 1e6)
	# y_tickslabels = [str(lab) for lab in y_tickslabels]
	# ax_win.set_yticks(y_ticks)
	# ax_win.set_yticklabels(y_tickslabels)

	# make vertical line befor experiments with loss>0.17
	tmp = [ind for ind, loss in enumerate(sorted_losses) if loss >= 0.17]

	ax_loss.axvline(tmp[0]-0.5, linestyle="--", lw=1)
	ax.axvline(tmp[0]-0.5, linestyle="--", lw=1)
	ax_win.axvline(tmp[0]-0.5, linestyle="--", lw=1)

	# plotly_fig = tls.mpl_to_plotly(mpl_fig)
	#
	# # For Legend
	# plotly_fig["layout"]["showlegend"] = True
	# plotly_fig["data"][0]["name"] = "CNN param count"
	# plotly_fig["data"][1]["name"] = "LSTM param count"
	# plotly_fig["data"][2]["name"] = "Output param count"
	# plotlypy.iplot(plotly_fig, filename='stacked-bar-chart')
	1
