import numpy as np
import tensorflow as tf
from keras.datasets import cifar10, mnist

def squeeze(input, out_D):

	in_D = input.get_shape().as_list()[3]
	w    = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, in_D, out_D]))
	b    = tf.Variable(tf.zeros([1, 1, 1, out_D]))

	return A(tf.nn.conv2d(input, w, strides=(1,1,1,1), padding='VALID') + b)


def expand(input, e1x1, e3x3):

	in_D = input.get_shape().as_list()[3]

	w1x1 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, in_D, e1x1]))
	b1x1 = tf.Variable(tf.zeros([1, 1, 1, e1x1]))
	c1x1 = A(tf.nn.conv2d(input, w1x1, strides=(1,1,1,1), padding='VALID') + b1x1)

	w3x3 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, in_D, e3x3]))
	b3x3 = tf.Variable(tf.zeros([1, 1, 1, e3x3]))
	c3x3 = A(tf.nn.conv2d(input, w3x3, strides=(1, 1, 1, 1), padding='SAME') + b3x3)

	return tf.concat([c1x1, c3x3], axis = 3)


def fire(input, s1x1, e1x1, e3x3):

	squeezed = squeeze(input, s1x1)
	expanded = expand(squeezed, e1x1, e3x3)

	return expanded


def model(in_H, in_W, in_D, classes, pool_size=(1,3,3,1)):

	train = tf.placeholder(tf.bool, shape=())
	x     = tf.placeholder(tf.float32, shape=[None,in_H,in_W,in_D])
	y     = tf.placeholder(tf.int32, shape=[None,1])

	w1    = tf.Variable(tf.contrib.layers.xavier_initializer()([7, 7, in_D, 96]))
	b1    = tf.Variable(tf.zeros([1, 1, 1, 96]))
	c1    = A(tf.nn.conv2d(x, w1, strides=(1,2,2,1), padding='VALID') + b1)

	maxp1 = tf.nn.max_pool(c1, ksize=pool_size, strides=(1,2,2,1), padding='VALID')

	fire2 = fire(maxp1, 16, 64, 64)
	fire3 = fire(fire2, 16, 64, 64)
	fire4 = fire(fire3, 32, 128, 128)

	maxp4 = tf.nn.max_pool(fire4, ksize=pool_size, strides=(1,2,2,1), padding='VALID')

	fire5 = fire(maxp4, 32, 128, 128)
	fire6 = fire(fire5, 48, 192, 192)
	fire7 = fire(fire6, 48, 192, 192)
	fire8 = fire(fire7, 64, 256, 256)

	maxp8 = tf.nn.max_pool(fire8, ksize=pool_size, strides=(1,2,2,1), padding='VALID')

	fire9 = fire(maxp8, 64, 256, 256)

	drop9 = tf.cond(train, lambda: tf.nn.dropout(fire9, keep_prob = KEEP_PROB), lambda: fire9)

	w10   = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, 512, classes]))
	b10   = tf.Variable(tf.zeros([1, 1, 1, classes]))
	c10   = A(tf.nn.conv2d(drop9, w10, strides=(1,1,1,1), padding='VALID') + b10)

	h, w  = c10.get_shape().as_list()[1:3]
	avgp  = tf.nn.avg_pool(c10, ksize=(1,h,w,1), strides=(1,1,1,1), padding='VALID')
	logs  = tf.squeeze(avgp, axis=[1,2])

	oneH  = tf.one_hot(y, classes)
	loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = oneH, logits = logs))
	opt   = tf.train.AdamOptimizer(LR, beta1 = MOMENTUM).minimize(loss)

	pred  = tf.reshape(tf.argmax(tf.nn.softmax(logs), axis=1, output_type = tf.int32), [-1, 1])
	acc   = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype = tf.float32))

	return x, y, train, loss, acc, opt


def whiten_data(data, mu = None, sigma = None):

	mu    = mu    if mu    else np.mean(data)
	sigma = sigma if sigma else np.std(data)
	data  = data - mu
	data  = data / sigma

	return data, mu, sigma


def load_data(d):

	if d is "cifar10":
		trn, tst = cifar10.load_data()
		return (32,32,3), 10, trn, tst

	elif d is "mnist":
		(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
		x_trn = np.expand_dims(x_trn, -1)
		x_tst  = np.expand_dims(x_tst, -1)
		y_trn = np.expand_dims(y_trn, -1)
		y_tst  = np.expand_dims(y_tst, -1)
		return (28,28,1), 10, (x_trn, y_trn), (x_tst, y_tst)

	elif d is "imagenet":
		raise NotImplementedError("ImageNet not implemented yet...")

	else:
		raise NotImplementedError("Unsupported data set: {}".format(d))


def run(iterations, batch_size=128, dataset="mnist", lr=4e-4, xavier=True, keep_prob=.6, momentum=.9, activation=tf.nn.relu):

	global LR, XAVIER, KEEP_PROB, MOMENTUM, A

	LR, XAVIER, KEEP_PROB, MOMENTUM, A = lr, xavier, keep_prob, momentum, activation
	print("Data: {}\nLR: {}\nKeep: {}\nMomentum: {}".format(dataset, LR, KEEP_PROB, MOMENTUM))

	(in_H, in_W, in_D), classes, (x_trn, y_trn), (x_tst, y_tst) = load_data(dataset)

	x_trn, mu_trn, sigma_trn = whiten_data(x_trn)
	x_tst = whiten_data(x_tst, mu_trn, sigma_trn)[0]

	x, y, train, loss, acc, opt = model(in_H, in_W, in_D, classes, (1,2,2,1))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in range(iterations):
		start  = np.random.randint(0, x_trn.shape[0] - batch_size)
		stop   = start + batch_size
		batch  = x_trn[start:stop, :, :, :]
		labels = y_trn[start:stop, :]

		_loss, _acc = sess.run([loss, acc, opt], feed_dict = { x:batch, y:labels, train:True })[:2]

		if i % 100 == 0:
			tst_acc = sess.run(acc, feed_dict = { x:x_tst, y:y_tst, train:False })
			print("Iter: {}\tLoss: {:.3f}\t Train: {:.3f}\tTest: {:.3f}".format(i, _loss, _acc, tst_acc))


run(50001, activation=tf.nn.leaky_relu)
