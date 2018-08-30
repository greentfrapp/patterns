import tensorflow as tf
import numpy as np
from PIL import Image
import imageio
from absl import flags
from absl import app
from time import strftime

from CPPN import CPPN

FLAGS = flags.FLAGS

flags.DEFINE_bool('gif', False, 'Generate a gif')

# model parameters
flags.DEFINE_integer('z', 2, 'z-dimension')
flags.DEFINE_float('scale', 1e1, 'Scale')
flags.DEFINE_list('units', [32, 32, 32], 'Units of each hidden layer')
flags.DEFINE_bool('use_r', True, 'Use radius input')

# gif parameters
flags.DEFINE_bool('loop', True, 'Loop for endless gif')
flags.DEFINE_integer('frames', 200, 'Number of frames')
flags.DEFINE_integer('fps', 30, 'Frames per second')

# output parameters
flags.DEFINE_integer('size', 200, 'Height/width of output in pixels')
flags.DEFINE_integer('height', None, 'Height of output in pixels')
flags.DEFINE_integer('width', None, 'Height of output in pixels')
flags.DEFINE_enum('type', 'L', ['L', 'RGB', 'HSV', 'RGBA', 'CMYK'], 'Output type - L, RGB, HSV, RGBA, CMYK')
flags.DEFINE_string('filename', None, 'Output filename')


def main(unused_args):

	z_dim = max(1, FLAGS.z)
	if FLAGS.loop:
		# See below on generating seamless loop
		z_dim = max(2, z_dim)	

	generator = CPPN(z_dim=z_dim, output_dim=len(FLAGS.type), scale=FLAGS.scale, units=FLAGS.units, use_r=FLAGS.use_r)

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		height = FLAGS.height or FLAGS.size
		width = FLAGS.width or FLAGS.size

		# Set ranges so that center of image is [0, 0]
		range_x = [-width // 2, width - width // 2 -1]
		range_y = [-height // 2, height - height // 2 -1]

		x = np.arange(range_x[0], range_x[1] + .5, 1)
		y = np.arange(range_y[0], range_y[1] + .5, 1)

		# Generate entire list of coordinates which will be passed to the model as a single batch
		coordinates = np.array([[x[i], y[j]] for i in range(len(x)) for j in range(len(y))])
		
		if FLAGS.gif:
			images = []
			frames = FLAGS.frames
			if FLAGS.loop:
				# See below on generating seamless loop
				if z_dim > 2:
					initial_z = np.random.uniform(-1.0, 1.0, size=(1, z_dim - 2)).astype(np.float32)
			else:
				initial_z = np.random.uniform(-1.0, 1.0, size=(1, z_dim)).astype(np.float32)
				end_z = np.random.uniform(-1.0, 1.0, size=(1, z_dim)).astype(np.float32)
		else:
			frames = 1
			z = np.random.uniform(-1.0, 1.0, size=(1, z_dim)).astype(np.float32)

		for i in np.arange(frames):
			if FLAGS.gif:
				if FLAGS.loop:
					# To get a nice seamless loop for the gif, we traverse a circle with two of the z dimensions
					z = np.array([np.cos(i*2*np.pi/frames), np.sin(i*2*np.pi/frames)]).reshape(1, 2)
					if z_dim > 2:
						z = np.concatenate([z, initial_z], axis=1)
				else:
					z = initial_z + (end_z - initial_z) * i / frames

			feed_dict = {
				generator.input_z: np.tile(z, (len(coordinates), 1)),
				generator.input_x: np.expand_dims(coordinates[:, 0], axis=1) / max(range_x),
				generator.input_y: np.expand_dims(coordinates[:, 1], axis=1) / max(range_y),
			}

			values = sess.run(generator.outputs, feed_dict)
			if FLAGS.type == 'L':
				values = values.reshape(len(y), len(x))
			else:
				values = values.reshape(len(y), len(x), len(FLAGS.type))
			values = (values * 255).astype(np.uint8)

			img = Image.fromarray(values, mode=FLAGS.type).convert('RGBA')

			if FLAGS.gif:
				images.append(np.array(img))
			else:
				img.save('{}.png'.format(FLAGS.filename or 'sample{}'.format(strftime('_%y%m%d_%H%M'))))

		if FLAGS.gif:
			imageio.mimsave('{}.gif'.format(FLAGS.filename or 'sample{}'.format(strftime('_%y%m%d_%H%M'))), images, duration=1./FLAGS.fps)


if __name__ == '__main__':
	app.run(main)
