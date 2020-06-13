#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


class C3DModel(object):
    def __init__(self, config):
        self._config = config

        self._input_data = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                self._config.batch_size,
                self._config.time_dimen,
                self._config.frame_height,
                self._config.frame_width,
                self._config.frame_channels
            ],
            name="input_video_segment")

        self._ground_truth = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                self._config.batch_size,
                self._config.ncls
            ],
            name="ground_truth")

        self._global_step = tf.Variable(0, trainable=False)
        self._optimizer, self._summary_op, self._probs = None, None, None
        self._loss, self._accuracy = None, None

        self.build_model()

    def build_model(self):
        """
        build U-Net model
        :return:
        """
        def conv3d(x, filters, activation, name):
            return tf.keras.layers.Conv3D(
                filters=filters,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='SAME',
                data_format='channels_last',
                activation=activation,
                name=name
            )(x)

        def pool3d(x, ksize, strides, name):
            return tf.nn.max_pool3d(
                input=x,
                ksize=ksize,
                strides=strides,
                padding='SAME',
                data_format="NDHWC",
                name=name)

        with tf.compat.v1.variable_scope("C3D-Backbone"):
            conv1 = conv3d(x=self._input_data, filters=64, activation=tf.nn.relu, name='conv1')
            relu1 = tf.nn.relu(conv1, 'relu1')
            pool1 = pool3d(x=relu1, ksize=(1, 2, 2), strides=(2, 2, 2), name='pool1')

            conv2 = conv3d(x=pool1, filters=96, activation=tf.nn.relu, name='conv2')
            relu2 = tf.nn.relu(conv2, 'relu2')
            pool2 = pool3d(x=relu2, ksize=(2, 2, 2), strides=(2, 2, 2), name='pool2')

            conv3 = conv3d(x=pool2, filters=128, activation=tf.nn.relu, name='conv3a')
            relu3a = tf.nn.relu(conv3, 'relu3a')
            conv3b = conv3d(x=relu3a, filters=128, activation=tf.nn.relu, name='conv3b')
            relu3b = tf.nn.relu(conv3b, 'relu3b')
            pool3 = pool3d(x=relu3b, ksize=(2, 2, 2), strides=(2, 2, 2), name='pool3')

            conv4 = conv3d(x=pool3, filters=256, activation=tf.nn.relu, name='conv4a')
            relu4a = tf.nn.relu(conv4, 'relu4a')
            conv4b = conv3d(x=relu4a, filters=256, activation=tf.nn.relu, name='conv4b')
            relu4b = tf.nn.relu(conv4b, 'relu4b')
            pool4 = pool3d(x=relu4b, ksize=(2, 2, 2), strides=(2, 2, 2), name='pool4')

            conv5 = conv3d(x=pool4, filters=256, activation=tf.nn.relu, name='conv5a')
            relu5a = tf.nn.relu(conv5, 'relu5a')
            conv5b = conv3d(x=relu5a, filters=256, activation=tf.nn.relu, name='conv5b')
            relu5b = tf.nn.relu(conv5b, 'relu5b')
            pool5 = pool3d(x=relu5b, ksize=(2, 2, 2), strides=(2, 2, 2), name='pool5')

            # Fully connected layer
            flatten = tf.reshape(pool5, shape=(self._config.batch_size, -1))
            fc1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
            fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=fc2, units=self._config.ncls, activation=None)
            self._probs = tf.nn.softmax(logits=logits)

        with tf.compat.v1.variable_scope("Loss"):
            self._loss = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(onehot_labels=self._ground_truth, logits=logits)
            )
            self._accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self._ground_truth, 1), tf.argmax(self._probs, 1)), dtype=tf.float32))

            tf.compat.v1.summary.scalar('loss', self._loss)

        with tf.compat.v1.variable_scope("optimization"):
            train_op = tf.compat.v1.train.AdamOptimizer(self._config.learning_rate)
            self._optimizer = train_op.minimize(self._loss)
            self._summary_op = tf.compat.v1.summary.merge_all()

    @property
    def loss(self):
        return self._loss

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def input_data(self):
        return self._input_data

    @property
    def probs(self):
        return self._probs

    @property
    def accuracy(self):
        return self._accuracy

