import tensorflow as tf

slim = tf.contrib.slim

# n = 9, kernel 3X3, 4 layers
def create_network(inputs, is_training, scope="win9_dep4", reuse=False):
	num_maps = 64
	kw = 3
	kh = 3

	with tf.variable_scope(scope, reuse=reuse):
		with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu, 
			normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
			
			net = slim.conv2d(inputs, num_maps, [kh, kw], scope='conv_bn_relu1')
			net = slim.repeat(net, 2, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu2_3')
			net = slim.conv2d(net, num_maps, [kh, kw], scope='conv4', activation_fn=None, 
					normalizer_fn=None)
			net = slim.batch_norm(net, is_training=is_training)

	return net



