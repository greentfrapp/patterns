import tensorflow as tf
import numpy as np
from PIL import Image
import imageio
from absl import flags
from absl import app
from time import strftime
from keras.datasets import mnist

from CPPN import CPPN


class Discriminator():

	def __init__(self, inputs):
		self.inputs = inputs

	def build_model():
		inputs = tf.reshape(self.inputs, [-1, 28, 28, 1])
		conv_1 = tf.layers.conv2d(
			inputs=self.inputs,
			filters=16,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='valid',
			activation=tf.nn.relu,
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_1',
		)
		maxpool_1 = tf.layers.max_pooling2d(
			inputs=conv_1,
			pool_size=(2, 2),
			strides=(2, 2),
			padding="valid",
		)
		conv_2 = tf.layers.conv2d(
			inputs=conv_1,
			filters=32,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='valid',
			activation=tf.nn.relu,
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_2',
		)
		maxpool_2 = tf.layers.max_pooling2d(
			inputs=conv_2,
			pool_size=(2, 2),
			strides=(2, 2),
			padding="valid",
		)
		unroll = tf.reshape(conv_2, [-1, 6*6*32])
		self.outputs = tf.layers.dense(
			inputs=unroll,
			units=1,
			activation=tf.sigmoid,
			name='dense',
		)


def main(unused_args):

	real_samples = tf.placeholder(
		shape=(None, 28, 28),
		dtype=tf.float32,
		name='real_samples',
	)

	with tf.variable_scope('generator'):
		generator = CPPN(z_dim=16, output_dim=1, scale=10., use_r=False)
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		discriminator_fake = Discriminator(inputs=generator.outputs)
		discriminator_real = Discriminator(inputs=real_samples)
	score_fake = discriminator_fake.outputs
	score_real = discriminator_real.outputs

	generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_fake), logits=self.score_fake))
	discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_real), logits=self.score_real)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.score_fake), logits=self.score_fake))

	discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
	discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(discriminator_loss, var_list=discriminator_vars)
	generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
	generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(generator_loss, var_list=generator_vars)


	(x_train, y_train), _ = mnist.load_data()
	# Select only 1
	x_train = x_train[np.where(y_train == 1)]

	# Set ranges so that center of image is [0, 0]
	width = height = 28
	range_x = [-width // 2, width - width // 2 -1]
	range_y = [-height // 2, height - height // 2 -1]

	x = np.arange(range_x[0], range_x[1] + .5, 1)
	y = np.arange(range_y[0], range_y[1] + .5, 1)

	# Generate entire list of coordinates which will be passed to the model as a single batch
	coordinates = np.array([[x[i], y[j]] for i in range(len(x)) for j in range(len(y))])

	batchsize = 64
	start = 0
	end = batchsize

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in np.arange(1000):
		if end > len(x_train):
			start = 0
			end = batchsize
		minibatch_x = x_train[start:end]
		start, end = end, end + batchsize

		feed_dict = {
			real_samples: minibatch_x,
			generator.input_z: np.tile(np.random.randn(len(minibatch_x), 1, 16), (1, len(coordinates), 1)),
			generator.input_x: np.tile(np.expand_dims(coordinates[:, 0], axis=1) / max(range_x), (len(minibatch_x), 1, 1)),
			generator.input_y: np.tile(np.expand_dims(coordinates[:, 1], axis=1) / max(range_y), (len(minibatch_x), 1, 1)),,
		}
		_, d_loss, _, g_loss = sess.run([discriminator_optimizer, discriminator_loss, generator_optimizer, generator_loss], feed_dict)

		if i % 100 == 0:
			print('Step {} - Disc. Loss: {:.3f} - Gen. Loss: {:.3f}'.format(i, d_loss, g_loss))




