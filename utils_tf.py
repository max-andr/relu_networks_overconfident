import argparse
import tensorflow as tf
import models
import numpy as np


def f_activation(activation_type):
    if activation_type == 'softplus':
        softplus_alpha = 1.0
        return lambda x: 1 / softplus_alpha * tf.nn.softplus(softplus_alpha * x)
    elif activation_type == 'relu':
        return tf.nn.relu
    elif activation_type == 'sigmoid':
        return tf.sigmoid
    else:
        raise Exception('Unknown activation function')


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def eval_in_batches(X_tf, Y_tf, tensors, tensors_upd, sess, batch_iterator, flag_train_tf,
                    init_metric_vars, feed_dict):
    """Get all predictions for a dataset by running it in large batches."""
    sess.run(init_metric_vars)
    for batch_x, batch_y in batch_iterator:
        sess.run(tensors_upd, feed_dict={X_tf: batch_x, Y_tf: batch_y, flag_train_tf: False, **feed_dict})
    vals = sess.run(tensors)
    return vals


def norm(v, lp):
    if lp == 1:
        norms = (tf.reduce_sum(tf.abs(v), axis=[1, 2, 3]))[..., None, None, None]
    elif lp == 2:
        norms = (tf.reduce_sum(v ** 2, axis=[1, 2, 3]) ** (1 / 2.))[..., None, None, None]
    elif lp == np.inf:
        norms = (tf.reduce_max(tf.abs(v), axis=[1, 2, 3]))[..., None, None, None]
    else:
        raise ValueError('wrong lp')
    return norms


def lp_project(x_adv, x_orig, eps, lp):
    x_adv = tf.clip_by_value(x_adv, 0., 1.)
    delta = x_adv - x_orig
    if lp == 2:
        norm_delta = norm(delta, lp=2)
        delta = delta / norm_delta * tf.minimum(eps, norm_delta)
    elif lp == np.inf:
        delta = tf.clip_by_value(delta, -eps, eps)
    else:
        raise ValueError('wrong lp')
    return x_orig + delta


def gen_rubbish_uniform(x, y):
    x = tf.random_uniform(tf.shape(x))
    y = tf.ones_like(y) * 1.0 / tf.to_float(y.shape[1])
    return x, y


def gen_rubbish_permuted_images(x, y):
    def permute_each(x_img):
        if len(x_img.shape) == 3:
            channels = x_img.shape[2]
            x_flat = tf.reshape(x_img, [-1, channels])
        else:
            x_flat = tf.reshape(x_img, [-1])
        x_flat = tf.random_shuffle(x_flat)
        x_permuted = tf.reshape(x_flat, x_img.shape)
        return x_permuted

    x = tf.map_fn(permute_each, x)
    y = tf.ones_like(y) * 1.0 / tf.to_float(y.shape[1])
    return x, y


def rescale_to_zero_one(x):
    min_val = tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
    max_val = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
    x = (x - min_val) / (max_val - min_val)
    return x


def apply_proper_conv(x, full_kernel):
    n_pad = int(full_kernel.shape[0])
    paddings = [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]]
    x = tf.pad(x, paddings, "SYMMETRIC")
    x = tf.nn.conv2d(x, full_kernel, strides=[1, 1, 1, 1], padding="SAME")
    x = x[:, n_pad:-n_pad, n_pad:-n_pad, :]
    x = rescale_to_zero_one(x)
    return x


def gaussian_kernel(std: float):
    """Makes 2D gaussian Kernel for convolution."""
    size = 7
    mean = 0.0

    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def apply_random_lowpass(x):
    std = tf.random_uniform([1], 1.0, 2.5)[0]
    gauss_kernel = gaussian_kernel(std)
    if x.shape[3] == 1:
        full_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    else:  # if 3 colors
        zero_kernel = tf.zeros_like(gauss_kernel)
        kernel1 = tf.stack([gauss_kernel, zero_kernel, zero_kernel], axis=2)
        kernel2 = tf.stack([zero_kernel, gauss_kernel, zero_kernel], axis=2)
        kernel3 = tf.stack([zero_kernel, zero_kernel, gauss_kernel], axis=2)
        full_kernel = tf.stack([kernel1, kernel2, kernel3], axis=3)
    x = apply_proper_conv(x, full_kernel)
    return x

def get_loss(logits, y, max_conf_flag_tf):
    maxclass = tf.argmax(logits, axis=-1)
    loss_elementwise = tf.cond(max_conf_flag_tf,
                    lambda: -tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(maxclass, logits.shape[-1])),
                    lambda: tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
                   )
    return loss_elementwise

def elementwise_best(x, loss_elementwise, x_best, loss_best_elementwise):
    take_prev = tf.to_float(loss_best_elementwise >= loss_elementwise)
    take_curr = tf.to_float(loss_best_elementwise < loss_elementwise)
    loss_best_elementwise = take_prev * loss_best_elementwise + take_curr * loss_elementwise
    x_best = tf.reshape(take_prev, [tf.shape(x)[0], 1, 1, 1]) * x_best + tf.reshape(take_curr, [tf.shape(x)[0], 1, 1, 1]) * x
    return x_best, loss_best_elementwise

def gen_adv_main(model, x, y, lp, eps, n_iters, step_size, max_conf_flag_tf):
    logits_x = model.get_logits(x, flag_train=False)
    tf.get_variable_scope().reuse_variables()
    loss_x = get_loss(logits_x, y, max_conf_flag_tf)
    
    assert lp == np.inf, 'Currently, only l-infinity attack is supported.' # We experimented with other norms before, but eventually only the infinity pgd attack was used.
        
    starting_perturbation = tf.random_uniform(minval=0.0, maxval=1.0, shape=(tf.shape(x)[0],1,1,1))
    unif = starting_perturbation*tf.random_uniform(minval=-eps, maxval=eps, shape=tf.shape(x))
    #unif = 0.0  # to remove the random step
    start_adv = tf.clip_by_value(x + unif, 0., 1.)
    logits_start = model.get_logits(start_adv, flag_train=False)
    tf.get_variable_scope().reuse_variables()
    loss_start = get_loss(logits_start, y, max_conf_flag_tf)
    
    x_best_start, loss_best_start = elementwise_best(start_adv, loss_start, x, loss_x)

    initial_vars = [0, start_adv, x_best_start, loss_best_start]
    cond = lambda i, x_adv, x_best, loss_best: tf.less(i, n_iters)
    def body(i, x_adv, x_best, loss_best):
        # we never update BN averages during generation of adv. examples
        logits = model.get_logits(x_adv, flag_train=False)
        tf.get_variable_scope().reuse_variables()
        loss = get_loss(logits, y, max_conf_flag_tf)
        g, = tf.gradients(tf.reduce_sum(loss), x_adv)
        g = tf.sign(g)
        x_adv = tf.stop_gradient(lp_project(x_adv + step_size * g, x, eps, lp))
        logits_after_upd = model.get_logits(x_adv, flag_train=False)
        loss_after_upd = get_loss(logits_after_upd, y, max_conf_flag_tf)
        x_best, loss_best = elementwise_best(x_adv, loss_after_upd, x_best, loss_best)
        return i + 1, x_adv, x_best, loss_best

    _, x_adv, x_best, _ = tf.while_loop(cond, body, initial_vars, back_prop=False,
                                        parallel_iterations=1)
    return tf.stop_gradient(x_best), y


def gen_adv(model, x, y, lp, eps, n_iters, step_size, rub_flag_tf, adv_flag_tf, frac_perm, apply_lowpass, max_conf_flag_tf):
    n_permuted = tf.to_int32(tf.to_float(tf.shape(x)[0]) * frac_perm)
    x_permuted, y_permuted = gen_rubbish_permuted_images(x[:n_permuted], y[:n_permuted])
    x_uniform, y_uniform = gen_rubbish_uniform(x[n_permuted:], y[n_permuted:])
    x_rubbish = tf.concat([x_permuted, x_uniform], axis=0)
    y_rubbish = tf.concat([y_permuted, y_uniform], axis=0)

    if apply_lowpass:
        x_rubbish = apply_random_lowpass(x_rubbish)
        
    # That's the main difference between adv. training and rubbish adversarial training - we start from noise
    x, y = tf.cond(rub_flag_tf,
                   lambda: (x_rubbish, y_rubbish),
                   lambda: (x, y))
    x_adv, y_adv = tf.cond(tf.logical_and(tf.logical_and(tf.greater(tf.shape(x)[0], 0),
                                                         adv_flag_tf),
                                          tf.greater(n_iters, 0)),
                           lambda: gen_adv_main(model, x, y, lp, eps, n_iters, step_size, max_conf_flag_tf),
                           lambda: (x, y))
    return x_adv, y_adv