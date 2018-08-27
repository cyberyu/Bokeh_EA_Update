from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, Panel, FuncTickFormatter, SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,Tabs, CheckboxButtonGroup, TableColumn, DataTable, Select, Button, TextInput)
from bokeh.layouts import column, row, WidgetBox
from bokeh.palettes import Category20_16
import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs
import pickle
import random
import math


old_data={}



def mean_standardize (d, m, sigma):
	return (d-m)/sigma

def inv_standardize (ds, m, sigma):
	return ds*sigma+m

@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles):
	with zs.BayesianNet(observed=observed) as model:

		ws = []
		for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
											  layer_sizes[1:])):
			w_mu = tf.zeros([1, n_out, n_in + 1])
			ws.append(
				zs.Normal('w' + str(i), w_mu, std=1.,n_samples=n_particles, group_ndims=2))

		# forward
		ly_x = tf.expand_dims(
			tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
		for i in range(len(ws)):
			w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
			ly_x = tf.concat(
				[ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
			ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.to_float(tf.shape(ly_x)[2]))
			if i < len(ws) - 1:
				ly_x = tf.nn.relu(ly_x)

		y_mean = tf.squeeze(ly_x, [2, 3])
		y_logstd = tf.get_variable('y_logstd', shape=[], initializer=tf.constant_initializer(0.))
		y = zs.Normal('y', y_mean, logstd=y_logstd)

	return model, y_mean

def density_tab():

	def construct_test_data(fund_in, risk_in, income_in, fga_in, contribue_in,  age_in,  retirementage_in,  planning_in, mean_x_train, std_x_train):

		switcher = {
			"Very Conservative": 0,
			"Conservative": 1,
			"Moderate": 2,
			"Aggressive": 3,
			"Very Aggressive":4}

		def risk_to_num(argument):
			func = switcher.get(argument, "nothing")
			return func

		y_to_retire = retirementage_in - age_in
		x = np.array([[math.log(fund_in), risk_to_num(risk_in), math.log(income_in), math.log(fga_in), math.log(contribue_in), age_in, y_to_retire, planning_in]*1]*2)  # double it cause some tech issue
		x_test_con = (x - mean_x_train)/std_x_train

		return x_test_con



	def make_plot_data(h):
		my_path = os.path.abspath(os.path.dirname(__file__))

		x_train_standardized = pickle.load(open(os.path.join(my_path, "x_train.pkl"), "rb"))
		y_train_standardized = pickle.load(open(os.path.join(my_path, "y_train.pkl"), "rb"))

		#x_test_standardized = pickle.load(open(os.path.join(my_path, "x_test.pkl"), "rb"))
		y_test_standardized = pickle.load(open(os.path.join(my_path, "y_test.pkl"), "rb"))
		mean_x_train = pickle.load(open(os.path.join(my_path, "mean_x_train.pkl"), "rb"))
		std_x_train = pickle.load(open(os.path.join(my_path, "std_x_train.pkl"), "rb"))
		mean_y_train = pickle.load(open(os.path.join(my_path, "mean_y_train.pkl"), "rb"))
		std_y_train = pickle.load(open(os.path.join(my_path, "std_y_train.pkl"), "rb"))

		model_file_path = 'modelsave/'


		my_path = os.path.abspath(os.path.dirname(__file__))
		exec(open(os.path.join(my_path, "predict.py")).read())


		# collect values to construct test data

		income_in = math.log(float(income_input.value))
		fga_in = math.log(float(fga_input.value))
		retirementage_in = int(retirementage_select.value)
		fund_in = math.log(float(fund_input.value))
		planning_in=int(planning_select.value)
		risk_in=risk_select.value
		age_in = int(age_select.value)
		contribute_in = math.log(float(contribution_input.value))


		print ('income_in ' + str(income_in))
		print('fga_in ' + str(fga_in))
		print('retirementage_in ' + str(retirementage_in))
		print('fund_in ' + str(fund_in))
		print('planning_in ' + str(planning_in))
		print('risk_select_in ' + str(risk_in))
		print('age_in ' + str(age_in))
		print('contribute_in ' + str(contribute_in))


		x_test_standardized = construct_test_data(fund_in, risk_in, income_in, fga_in, contribute_in, age_in, retirementage_in, planning_in, mean_x_train, std_x_train)

		y_test_pred_mean, y_test_pred_all= predict(model_file_path, x_train_standardized, y_train_standardized, x_test_standardized, y_test_standardized, mean_y_train, std_y_train, h)

		y_test_pred_mean = y_test_pred_mean[0]
		y_test_pred_all = y_test_pred_all[:,0]

		# denormalize and recover the true value
		y_true = y_test_standardized[0]*std_y_train+mean_y_train

		# # denormalize and recover the mean predicted value
		#
		# y_test_pred_mean_denormalized = y_test_pred_mean[0]*std_y_train+mean_y_train
		#
		# # denormalize and recover the all predicted values
		#
		# y_test_pred_all_denormalized = y_test_pred_all[0]*std_y_train + mean_y_train


		# calculate the prediciton variance

		y_var_mat = y_test_pred_all - y_test_pred_mean
		y_var_mat_square = np.square(y_var_mat)
		y_var_mat_square_mean = np.sqrt(np.mean(y_var_mat_square, 0))


		print ('y_test_pred_mean shape ' + str(y_test_pred_mean))
		print ('y_var_mat shape ' + str(y_var_mat.shape))
		print('y_var_mat_square shape ' + str(y_var_mat_square.shape))
		print('y_var_mat_square_mean shape ' + str(y_var_mat_square_mean.shape))

		# plot the histogram of y_test_pred_all_denormalized
		arr_hist, edges = np.histogram(y_test_pred_all, density=True, bins=20)
		#arr_hist = arr_hist/np.sum(arr_hist)

		print ('arr_hist value is ' + str(arr_hist) + ' and sum is '+ str(np.sum(arr_hist)))

		by_carrier = pd.DataFrame(columns=['proportion', 'left', 'right','f_proportion', 'f_interval','name', 'color'])

		arr_df = pd.DataFrame({'proportion': arr_hist, 'left': edges[:-1], 'right': edges[1:]})

		# Format the proportion
		arr_df['f_proportion'] = ['%0.5f' % proportion for proportion in arr_df['proportion']]

		# Format the interval
		arr_df['f_interval'] = ['%d to %d minutes' % (left, right) for left, right in
								zip(arr_df['left'], arr_df['right'])]

		# Assign the carrier for labels
		arr_df['name'] = 'Retirement_'+str(h) + ' success rate ' + str(y_test_pred_mean)
		arr_df['color'] = Category20_16[h]

		by_carrier=by_carrier.append(arr_df)

		if ('foo' in old_data):
			by_carrier = by_carrier.append(old_data['foo'])
			old_data['foo'] = by_carrier



		# construct density plot

		x_lin = np.linspace(-50, 120, 1000)
		pdf = 1 / (y_var_mat_square_mean * np.sqrt(2 * np.pi)) * np.exp(-(x_lin - y_test_pred_mean) ** 2 / (2 * y_var_mat_square_mean ** 2))

		xs_pdf = []
		ys_pdf = []
		colors_pdf = []
		labels_pdf = []

		if ('pdf' in old_data):
			old_pdf_df = old_data['pdf']

			x_old = old_pdf_df['x']
			y_old = old_pdf_df['y']
			col_old = old_pdf_df['color']
			lab_old = old_pdf_df['label']

			# pickle.dump(x_old, open("x_old.pkl", "wb"))
			# pickle.dump(y_old, open("y_old.pkl", "wb"))
			# pickle.dump(col_old, open("col_old.pkl", "wb"))
			# pickle.dump(lab_old, open("lab_old.pkl", "wb"))
			#
			# pickle.dump(x_lin, open("x_lin.pkl", "wb"))
			# pickle.dump(pdf, open("pdf.pkl", "wb"))
			#pickle.dump(Category20_16[h], open("col.pkl", "wb"))
			#pickle.dump('Retirement_'+str(h) + '_pdf Sigma '+ str(y_var_mat_square_mean), open("lab.pkl", "wb"))
			#
			# print('Inside pdf data x_old ' + str(len(x_old[0])) + ' Type ' + str(type(x_old[0])))



			# xs_pdf = xs_pdf.append(x_old[0])
			# ys_pdf = ys_pdf.append(y_old[0])
			#
			#
			#
			# colors_pdf = colors_pdf.append(col_old)
			# labels_pdf = labels_pdf.append(lab_old)
			print('Inside pdf data col_old ' + str(len(col_old)) + ' Type ' + str(type(col_old)))
			print('Inside pdf data lab_old ' + str(len(lab_old)) + ' Type ' + str(type(lab_old)))

			x_old.append(list(x_lin))
			y_old.append(list(pdf))
			col_old.append(Category20_16[h])
			lab_old.append('Retirement_'+str(h) + '_pdf Sigma '+ str(y_var_mat_square_mean))

			print('Inside pdf data x_old ' + str(len(x_old)) + ' Type ' + str(type(x_old)))
			print('Inside pdf data y_old ' + str(len(y_old)) + ' Type ' + str(type(y_old)))
			print('Inside pdf data col_old ' + str(len(col_old)) + ' Type ' + str(type(col_old)))
			print('Inside pdf data lab_old ' + str(len(lab_old)) + ' Type ' + str(type(lab_old)))



			# print('Inside pdf data xs_pdf ' + str(len(xs_pdf)) + ' Type ' + str(type(xs_pdf)))
			# print('Inside pdf data xs_pdf ' + str(len(xs_pdf[0])) + ' Type ' + str(type(xs_pdf[0])))
			# #
			# xs_pdf = xs_pdf.append(list(x_lin))
			# ys_pdf = ys_pdf.append(list(pdf))
			# colors_pdf = colors_pdf.append(Category20_16[h])
			# labels_pdf = labels_pdf.append('Retirement_'+str(h) + '_pdf Sigma '+ str(y_var_mat_square_mean))
			#
			# print('Inside pdf data xs_pdf ' + str(len(xs_pdf)) + ' Type ' + str(type(xs_pdf)))
			# print('Inside pdf data xs_pdf ' + str(len(xs_pdf[0])) + ' Type ' + str(type(xs_pdf[0])))
			#

			pdf_data = {'x': x_old, 'y': y_old, 'color': col_old, 'label': lab_old}

		else:

			xs_pdf.append(list(x_lin))
			ys_pdf.append(list(pdf))
			colors_pdf.append(Category20_16[h])
			labels_pdf.append('Retirement_' + str(h) + '_pdf Sigma ' + str(y_var_mat_square_mean))

			pdf_data = {'x': xs_pdf, 'y': ys_pdf, 'color': colors_pdf, 'label': labels_pdf}

		old_data['pdf'] = pdf_data

		pdf_src = ColumnDataSource(data=pdf_data)

		#return ColumnDataSource(by_carrier), arr_df, y_test_pred_mean, y_var_mat_square_mean, arr_hist, edges, pdf_src, pdf_data
		return ColumnDataSource(by_carrier), arr_df, pdf_src, pdf_data
		#return ColumnDataSource(by_carrier), arr_df, y_test_pred_mean, y_var_mat_square_mean


	def make_new_plot(h_src, p_src):
	#def make_new_plot(new_src):

		# print ('mu is ' + str(mu))
		# print ('sigma is ' + str(sig))

		p = figure(plot_width=1000, plot_height=700,
		 		   title='Predicted Success Rate real value ',
		 		   x_axis_label='Retirement Success Rate', y_axis_label='Density')

		p.quad(source=h_src, bottom=0, top='proportion', left='left', right='right',
			   color='color', fill_alpha=0.7, hover_fill_color='color', legend='name',
			   hover_fill_alpha=1.0, line_color='black')

		# p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
		# 		fill_color="#036564", line_color="#033649")


		p.multi_line('x', 'y', color='color', legend='label', line_width=3, source=p_src)
		#p.line(x_lin, pdf, line_color="black", line_width=8, alpha=0.7, legend="PDF")

		p = style(p)

		return p

	# def update(attr, old, new):
	# 	# List of carriers to plot
	# 	carriers_to_plot = [carrier_selection.labels[i] for i in
	# 						carrier_selection.active]
	#
	# 	# If no bandwidth is selected, use the default value
	# 	if bandwidth_choose.active == []:
	# 		bandwidth = None
	# 	# If the bandwidth select is activated, use the specified bandwith
	# 	else:
	# 		bandwidth = bandwidth_select.value
	#
	# 	new_src = make_dataset(carriers_to_plot,
	# 						   range_start=range_select.value[0],
	# 						   range_end=range_select.value[1],
	# 						   bandwidth=bandwidth)
	#
	# 	src.data.update(new_src.data)

	def style(p):
		# Title
		p.title.align = 'center'
		p.title.text_font_size = '20pt'
		p.title.text_font = 'serif'

		# Axis titles
		p.xaxis.axis_label_text_font_size = '14pt'
		p.xaxis.axis_label_text_font_style = 'bold'
		p.yaxis.axis_label_text_font_size = '14pt'
		p.yaxis.axis_label_text_font_style = 'bold'

		# Tick labels
		p.xaxis.major_label_text_font_size = '12pt'
		p.yaxis.major_label_text_font_size = '12pt'

		return p

	# def update(attr, old, new):
	# 	h = random.randint(1, 100)
	# 	print('New random number is ' + str(h))
	# 	nd = make_plot_data(h)
	# 	new_src.data.update(nd.data)

	def draw_simulation():
		h = random.randint(1, 10)

		print ('New random number is '+ str(h))
		hist_src, rdf, pdf_src, pdf_df= make_plot_data(h)

		#print ('pdf df ' + str(pdf_df))
		#nd,rdf, mean, sig, hist, edges, pdf_src, pdf_data = make_plot_data(h)

		h_src.data.update(hist_src.data)
		p_src.data.update(pdf_src.data)



	@zs.reuse('variational')
	def mean_field_variational(layer_sizes, n_particles):
		with zs.BayesianNet() as variational:
			ws = []
			for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
				w_mean = tf.get_variable('w_mean_' + str(i), shape=[1, n_out, n_in + 1],initializer=tf.constant_initializer(0.))
				w_logstd = tf.get_variable('w_logstd_' + str(i), shape=[1, n_out, n_in + 1], initializer=tf.constant_initializer(0.))
				ws.append(
					zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
							  n_samples=n_particles, group_ndims=2))
			return variational


	def predict(model_file_path, x_train_standardized, y_train_standardized, x_test_standardized, y_test_standardized, mean_y_train, std_y_train, h):


		tf.set_random_seed(1237)
		np.random.seed(1234)

		x_train = x_train_standardized
		y_train = y_train_standardized
		x_test = x_test_standardized
		y_test = y_test_standardized

		#x_test = x_test[h:h+1,:]
		y_test = y_test[h:h+1]

		N, n_x = x_train.shape

		print ('X test is ' + str(x_test))

		# with tf.variable_scope(tf.get_variable_scope(), reuse=True) as vscope:
		# 	#tf.get_variable_scope().reuse_variables()

		# Define model parameters
		n_hiddens = [20]

		# Build the computation graph
		n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
		x = tf.placeholder(tf.float32, shape=[None, n_x])
		y = tf.placeholder(tf.float32, shape=[None])
		layer_sizes = [n_x] + n_hiddens + [1]  # layer_sizes is

		w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

		def log_joint(observed):
			model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
			log_pws = model.local_log_prob(w_names)
			log_py_xw = model.local_log_prob('y')
			return tf.add_n(log_pws) + log_py_xw * N

		variational = mean_field_variational(layer_sizes, n_particles)
		qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
		latent = dict(zip(w_names, qw_outputs))
		lower_bound = zs.variational.elbo(
			log_joint, observed={'y': y}, latent=latent, axis=0)
		cost = tf.reduce_mean(lower_bound.sgvb())
		lower_bound = tf.reduce_mean(lower_bound)

		tf.global_variables_initializer()
		#optimizer = tf.train.GradientDescentOptimizer(0.01)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

		tf.global_variables_initializer()
		infer_op = optimizer.minimize(cost)

		# prediction: rmse & log likelihood
		observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
		observed.update({'y': y})
		model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
		y_pred = tf.reduce_mean(y_mean, 0)
		l2 = tf.norm(y_pred - y)/tf.norm(y)
		log_py_xw = model.local_log_prob('y')

		devs_squared = tf.square(y_pred-y_mean)
		var=tf.reduce_mean(devs_squared) #*std_y_train*std_y_train

		# Define training/evaluation parameters
		lb_samples = 10
		ll_samples = 2000
		epochs = 500
		batch_size = 10
		iters = int(np.floor(x_train.shape[0] / float(batch_size)))
		test_freq = 10

		# Add a train server object
		saver = tf.train.Saver(save_relative_paths=True)

		# Run the inference
		with tf.Session() as sess:

			#tf.get_variable_scope().reuse_variables()
			sess.run(tf.global_variables_initializer())
			#sess.run(tf.global_variables_initializer())
			# Restore from the latest checkpoint
			ckpt_file = tf.train.latest_checkpoint(model_file_path)
			begin_epoch = 1
			if ckpt_file is not None:
				print('Restoring model from {}...'.format(ckpt_file))
				begin_epoch = int(ckpt_file.split('.')[-2]) + 1
				saver.restore(sess, ckpt_file)

		#         for epoch in range(begin_epoch, epochs + 1):
		#             lbs = []
		#             for t in range(iters):
		#                 x_batch = x_train[t * batch_size:(t + 1) * batch_size]
		#                 y_batch = y_train[t * batch_size:(t + 1) * batch_size]
		#                 _, lb = sess.run(
		#                     [infer_op, lower_bound],
		#                     feed_dict={n_particles: lb_samples,
		#                                x: x_batch, y: y_batch})
		#                 lbs.append(lb)
		#             print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))

		#             if epoch % test_freq == 0:
			test_lb, y_test_pred, ne, pred_var, y_test_mean = sess.run([lower_bound, y_pred, l2, var, y_mean],feed_dict={n_particles: ll_samples,x: x_test, y: y_test})

			y_test_mean = (y_test_mean*std_y_train+mean_y_train)
			y_test_pred = (y_test_pred*std_y_train+mean_y_train)

			print ('Y test pred shape is ' + str(y_test_pred.shape))
			print('Y test mean shape is ' + str(y_test_mean.shape))
			# test_lb, test_rmse, test_ll, y_test_pred, y_test_var, y_test_mean = sess.run(
			# 	[lower_bound, rmse, log_likelihood, y_pred, var, y_mean],
			# 	feed_dict={n_particles: ll_samples, x: x_test, y: y_test})
			#
			#
			# l2_error = NA.norm((y_test_pred - y_test))/NA.norm(y_test)
			#
			# y_test_mean_n = inv_standardize(y_test_mean, mean_y_train, std_y_train)
			# y_test_pred_n = inv_standardize(y_test_pred, mean_y_train, std_y_train)

			# y_var_mat = y_test_mean - y_test_pred
			# y_var_mat_square = tf.square(y_var_mat)
			# y_var_mat_square_mean = tf.reduce_mean(y_var_mat_square, 0)

			print('>> TEST')
			print('>> Test lower bound = {}, l2_internal={}, pred_var={}'.format(test_lb, ne, pred_var))

		return y_test_pred, y_test_mean


	# FUNDS_EARMARKED_FOR_GOAL
	fund_input = TextInput(value="228899", title="Funds for Goal:")

	contribution_input = TextInput(value="14573", title="Contribution Amount:")

	income_input = TextInput(value="204898", title="Annual Salary:")

	fga_input = TextInput(value="6039273", title="Future Goal Amount:")

	retirementage_select = Slider(start = 20, end = 100, step = 1, value = 69, title = 'Projected Retirement Age')
	#retirementage_select.on_change('value', update)

	planning_select = Slider(start = 90, end = 120, step = 1, value = 100, title = 'Planning_Horizon')
	#planning_select.on_change('value', update)


	# Risk Tolerance
	risk_select = Select(title="Risk Tolerance:", value="Moderate", options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"])


	age_select = Slider(start = 20, end = 100, step = 1, value = 43, title = 'Age')
	#age_select.on_change('value', update)

	b = Button(label='Simulate')
	b.on_click(draw_simulation)

	# Make the density plot

	h = random.randint(1, 10)

	#new_src, rd, mean, sig = make_plot_data(h)

	h_src, rd, p_src, pdf_df = make_plot_data(h)

	#print ('new_src is ' + str(new_src))
	old_data['foo'] = rd

	#p2 = make_new_plot(new_src)

	p2 = make_new_plot(h_src, p_src)
	old_data['pdf'] = pdf_df

	# Add style to the plot
	p2 = style(p2)

	# Put controls in a single element
	controls = WidgetBox(fund_input, contribution_input, income_input, fga_input, retirementage_select, planning_select, risk_select, age_select, b)

	# Create a row layout
	layout = row(controls, p2)

	# Make a tab with the layout
	tab = Panel(child=layout, title='Density Plot')

	return tab


