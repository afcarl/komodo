import tensorflow as tf
import numpy as np

def dense(x, units, activation=None):
    input_dim = int(x.shape[-1])
    sd = 1.0 / np.sqrt(input_dim)
    weights = tf.Variable(tf.truncated_normal([input_dim, units], stddev=sd), name='weights')
    bias = tf.Variable(tf.constant(0.1, shape=[units]), name='bias')
    return  activation(tf.matmul(x, weights) + bias)

def mlp(x, sizes, activation, scope=''):
    with tf.variable_scope(scope):
        for size in sizes[:-1]:
            x = dense(x, units=size, activation=activation)
        if sizes[-1] == 1:
            return tf.squeeze(tf.layers.dense(x, units=sizes[-1]))
        else:
            return tf.layers.dense(x, units=sizes[-1])

def mlp(x, sizes, activation, scope=''):
    with tf.variable_scope(scope):
        for size in sizes[:-1]:
            x = tf.layers.dense(x, units=size, activation=activation)
        if sizes[-1] == 1:
            return tf.squeeze(tf.layers.dense(x, units=sizes[-1]))
        else:
            return tf.layers.dense(x, units=sizes[-1])

def cnn(x, output_dim, scope='', shared=False):
    if shared:
        with tf.variable_scope(scope):

            # conv1
            x = tf.layers.conv2d(
                inputs=x,
                filters=32,
                kernel_size=8,
                strides=4,
                padding='valid',
                activation=tf.nn.relu)

            # conv2
            x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=4,
                strides=2,
                padding='valid',
                activation=tf.nn.relu)

            # conv3
            x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=3,
                strides=1,
                padding='valid',
                activation=tf.nn.relu)

            # dense
            x = tf.layers.dense(tf.reshape(x, (-1, 64 * 7 * 7)), 512, tf.nn.relu)

            # value
            values = tf.squeeze(tf.layers.dense(x, 1))

            # policy
            policy_logits = tf.layers.dense(x, output_dim)

            return values, policy_logits
    else:
        with tf.variable_scope(scope):

            # conv1
            x = tf.layers.conv2d(
                inputs=x,
                filters=32,
                kernel_size=8,
                strides=4,
                padding='valid',
                activation=tf.nn.relu)

            # conv2
            x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=4,
                strides=2,
                padding='valid',
                activation=tf.nn.relu)

            # conv3
            x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=3,
                strides=1,
                padding='valid',
                activation=tf.nn.relu)

            # dense
            x = tf.layers.dense(tf.reshape(x, (-1, 64 * 7 * 7)), 512, tf.nn.relu)

            # output
            if output_dim == 1:
                return tf.squeeze(tf.layers.dense(x, output_dim))
            else:
                return tf.layers.dense(x, output_dim)

def clip_by_norm(gradients, clip_norm):
    new_gradients = []
    for g,v in gradients:
        if g is not None:
            g = tf.clip_by_norm(g, clip_norm=clip_norm)
        new_gradients += [(g,v)]
    return new_gradients
