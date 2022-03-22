import tensorflow as tf
import numpy as np


def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5 )
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5 )


def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear', mask=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
                            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if mask is not None:
            out = tf.minimum(out, (2.0 * mask - 1.0) * np.finfo(np.float32).max)

        if activation_fn is not None:
            return activation_fn(out), w, b
        else:
            return out, w, b
