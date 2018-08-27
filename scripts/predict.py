import zhusuan as zs
import tensorflow as tf
import os
import numpy as np

@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles):

    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([1, n_out, n_in + 1])
            ws.append(
                zs.Normal('w' + str(i), w_mu, std=1.,
                          n_samples=n_particles, group_ndims=2))

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

@zs.reuse('variational')
def mean_field_variational(layer_sizes, n_particles):
    with zs.BayesianNet() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            w_mean = tf.get_variable('w_mean_' + str(i), shape=[1, n_out, n_in + 1],
                                     initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable('w_logstd_' + str(i), shape=[1, n_out, n_in + 1], initializer=tf.constant_initializer(0.))
            ws.append(
                zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                          n_samples=n_particles, group_ndims=2))
        return variational


def predict(model_file_path, x_train_standardized, y_train_standardized, x_test_standardized, y_test_standardized,
        mean_y_train, std_y_train, h):

        tf.set_random_seed(1237)
        np.random.seed(1234)

        x_train = x_train_standardized
        y_train = y_train_standardized
        x_test = x_test_standardized
        y_test = y_test_standardized

        x_test = x_test[h:h+1, :]
        y_test = y_test[h:h+1]

        N, n_x = x_train.shape


        #tf.get_variable_scope().reuse_variables()

        # Define model parameters
        n_hiddens = [20]

        # # Build the computation graph
        # with tf.variable_scope(tf.get_variable_scope(), reuse=True) as vscope:

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

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        #optimizer= tf.train.GradientDescentOptimizer(0.01)
        tf.global_variables_initializer()

        infer_op = optimizer.minimize(cost)

        # prediction: rmse & log likelihood
        observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
        observed.update({'y': y})
        model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        y_pred = tf.reduce_mean(y_mean, 0)
        l2 = tf.norm(y_pred - y) / tf.norm(y)
        log_py_xw = model.local_log_prob('y')

        devs_squared = tf.square(y_pred - y_mean)
        var = tf.reduce_mean(devs_squared)  # *std_y_train*std_y_train

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

            sess.run(tf.global_variables_initializer())

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
            test_lb, y_test_pred, ne, pred_var, y_test_mean = sess.run([lower_bound, y_pred, l2, var, y_mean],
                                                                       feed_dict={n_particles: ll_samples, x: x_test,
                                                                                  y: y_test})

            y_test_mean = (y_test_mean * std_y_train + mean_y_train)
            y_test_pred = (y_test_pred * std_y_train + mean_y_train)

            print('Y test pred shape is ' + str(y_test_pred.shape))
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
