import tensorflow as tf


def mlp(x, sizes, activation, output_activation=None):
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    if sizes[-1] == 1:
        return tf.squeeze(tf.layers.dense(x, units=sizes[-1], activation=output_activation))
    else:
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def cnn(x, output_dim, scope=''):

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
