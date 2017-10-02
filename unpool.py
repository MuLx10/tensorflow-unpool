import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim



POOL_SIZE = 7



def max_pool_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax(net,ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1],padding='SAME')
    mask = tf.stop_gradient(mask)
    net = slim.max_pool2d(net, kernel_size=[stride, stride],  stride=POOL_SIZE)
    return net, mask





def unpool(net, mask, stride):
  	assert mask is not None
	ksize = [1, stride, stride, 1]
	input_shape = net.get_shape().as_list()
	#  calculation new shape
	output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
	# calculation indices for batch, height, width and feature maps
	one_like_mask = tf.ones_like(mask)
	batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
	b = one_like_mask * batch_range
	y = mask // (output_shape[2] * output_shape[3])
	x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
	feature_range = tf.range(output_shape[3], dtype=tf.int64)
	f = one_like_mask * feature_range
	# transpose indices & reshape update values to one dimension
	updates_size = tf.size(net)
	indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
	values = tf.reshape(net, [updates_size])
	ret = tf.scatter_nd(indices, values, output_shape)
	return ret






## Demonstration

# shape = [10, 28, 28, 1]


# input_tensor = np.random.random(shape)
# input_tensor = np.array(input_tensor,dtype='float32')
# input_tensor = tf.Variable(input_tensor,dtype=tf.float32)

# net = slim.conv2d(input_tensor, 16, [5, 5])
# net = slim.conv2d(net, 32, [3, 3])

# pooled_layer, mask = max_pool_with_argmax(net, POOL_SIZE)
# unpooled_layer = unpool(pooled_layer, mask, stride=POOL_SIZE)


# init = tf.global_variables_initializer()
# with tf.Session() as sess:
# 	sess.run(init)
# 	sess.run([input_tensor])
