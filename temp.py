import tensorflow as tf
import numpy as np
from PIL import Image
import imageio

input_z = tf.placeholder(
	shape=(None, 2),
	dtype=tf.float32,
	name='input_z',
)

input_x = tf.placeholder(
	shape=(None, 1),
	dtype=tf.float32,
	name='input_x',
)

input_y = tf.placeholder(
	shape=(None, 1),
	dtype=tf.float32,
	name='input_y',
)

input_r = (input_x ** 2 + input_y **2) ** 0.5

inputs = tf.concat([input_z, input_x, input_y, input_r], axis=1)

inputs *= 1e1

dense = inputs
for i in np.arange(3):
	dense = tf.layers.dense(
		inputs=dense,
		units=32,
		activation=tf.tanh,
		kernel_initializer=tf.random_normal_initializer(),
		name='dense_{}'.format(i),
	)

outputs = tf.layers.dense(
	inputs=dense,
	units=1,
	activation=tf.sigmoid,
	kernel_initializer=tf.random_normal_initializer(),
	name='outputs',
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

range_x = [-100, 100]
range_y = [-100, 100]

x = np.arange(range_x[0], range_x[1] + .5, 1)
y = np.arange(range_y[0], range_y[1] + .5, 1)

coordinates = np.array([[x[i], y[j]] for i in range(len(x)) for j in range(len(y))])

initial_z = np.random.uniform(-1.0, 1.0, size=(1, 6)).astype(np.float32)
end_z = np.random.uniform(-1.0, 1.0, size=(1, 8)).astype(np.float32)
delta = 0.005
images = []
frames = 200

for i in np.arange(frames):
	# z = initial_z + (end_z - initial_z) * i / frames
	z = np.array([np.cos(i*2*np.pi/frames), np.sin(i*2*np.pi/frames)]).reshape(1, 2)
	# z = np.concatenate([z, z, z, z], axis=1)
	feed_dict = {
		input_z: np.tile(z, (len(coordinates), 1)),
		input_x: np.expand_dims(coordinates[:, 0], axis=1) / max(range_x),
		input_y: np.expand_dims(coordinates[:, 1], axis=1) / max(range_y),
	}

	values = sess.run(outputs, feed_dict).reshape(len(x), len(y))
	values = (values * 255).astype(np.uint8)

	# Image.fromarray(values).convert('RGBA').save('samples/test/{}.png'.format(i))
	img = Image.fromarray(values, mode='L').convert('RGBA')

	# images.append(imageio.imread('samples/{}.png'.format(i)))
	images.append(np.array(img))

	# z += np.random.uniform(-delta, delta, size=(1, 8)).astype(np.float32)

imageio.mimsave('cppn.gif', images, duration=0.05)
