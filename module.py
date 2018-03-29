import tensorflow as tf


def sequence_mask(lens, max_len):
    len_t = tf.expand_dims(lens, 1)
    range_t = tf.range(0, max_len, 1)
    range_row = tf.expand_dims(range_t, 0)
    mask = tf.cast(tf.less(range_row, len_t), tf.float32)
    return mask


def layer_normalization(inputs, epsilon=1e-8, scope='ln', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [2], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert(False)
    return optfn