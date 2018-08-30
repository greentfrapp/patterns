import tensorflow as tf

class CPPN():

	def __init__(self, z_dim=2, output_dim=1, scale=1e1, units=[32, 32, 32], activations=None, use_r=True):
		self.z_dim = z_dim
		self.output_dim = output_dim
		self.scale = scale
		self.units = units
		self.activations = activations or [tf.tanh] * len(units)
		if len(self.activations) != len(self.units):
			raise ValueError('Length of activations is not the same as length of units')
		self.use_r = use_r
		self.build_model()

	def build_model(self):

		self.input_z = tf.placeholder(
			shape=(None, self.z_dim),
			dtype=tf.float32,
			name='input_z',
		)

		self.input_x = tf.placeholder(
			shape=(None, 1),
			dtype=tf.float32,
			name='input_x',
		)

		self.input_y = tf.placeholder(
			shape=(None, 1),
			dtype=tf.float32,
			name='input_y',
		)

		if self.use_r:
			input_r = (self.input_x ** 2 + self.input_y **2) ** 0.5
			inputs = tf.concat([self.input_z, self.input_x, self.input_y, input_r], axis=1)
		else:
			inputs = tf.concat([self.input_z, self.input_x, self.input_y], axis=1)

		inputs *= self.scale

		dense = inputs
		for i, n_units in enumerate(self.units):
			dense = tf.layers.dense(
				inputs=dense,
				units=n_units,
				activation=self.activations[i],
				kernel_initializer=tf.random_normal_initializer(),
				name='dense_{}'.format(i),
			)

		self.outputs = tf.layers.dense(
			inputs=dense,
			units=self.output_dim,
			activation=tf.sigmoid,
			kernel_initializer=tf.random_normal_initializer(),
			name='outputs',
		)