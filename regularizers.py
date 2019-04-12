import tensorflow as tf


def weight_decay(norm=2):
    """
    L2 weight decay loss, based on all weights that have var_pattern in their name

    var_pattern - a substring of a name of weights variables that we want to use in Weight Decay.
    """
    costs = []
    for var in tf.trainable_variables():
        if 'weight' in var.op.name or 'fc' in var.op.name or 'conv' in var.op.name:
            if norm == 1:
                lp_norm_var = tf.reduce_sum(tf.abs(var))
            elif norm == 2:
                lp_norm_var = tf.reduce_sum(tf.square(var))
            else:
                raise ValueError('wrong norm of weight decay')
            costs.append(lp_norm_var)
    return tf.add_n(costs)

