import tensorflow as tf

slim = tf.contrib.slim

# n = 29, 5 layers of kernel 5X5, 4 layers of kernel 3X3, 9 layers in total
def create_network(inputs, is_training, scope="win29_dep9", reuse=False):
	num_maps = 64
	kw1 = 3
	kh1 = 3
    kw2 = 5
	kh2 = 5

	with tf.variable_scope(scope, reuse=reuse):
		with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu, 
			normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
			
			net = slim.conv2d(inputs, num_maps, [kh2, kw2], scope='conv_bn_relu1')
			net = slim.repeat(net, 4, slim.conv2d, num_maps, [kh2, kw2], scope='conv_bn_relu2_5')
            net = slim.conv2d(net, num_maps, [kh1, kw1], scope='conv_bn_relu6')
            net = slim.repeat(net, 2, slim.conv2d, num_maps, [kh1, kw1], scope='conv_bn_relu2_8')
			net = slim.conv2d(net, num_maps, [kh1, kw1], scope='conv9', activation_fn=None, 
					normalizer_fn=None)
			net = slim.batch_norm(net, is_training=is_training)

	return net



