import numpy as np
import tensorflow as tf
from keras.datasets import cifar10, mnist

def squeeze(input, channels, layer_num):
    """
    Defines squeezed block for fire module.

    :param input: input tensor
    :param channels: number of output channels
    :param layer_num: layer number for naming purposes
    :return: output tensor convoluted with squeeze layer
    """
    layer_name = 'squeeze_' + layer_num
    input_channels = input.get_shape().as_list()[3]

    with tf.name_scope(layer_name):
        weights = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, input_channels, channels]))
        biases = tf.Variable(tf.zeros([1, 1, 1, channels]), name = 'biases')
        nets = tf.nn.conv2d(input, weights, strides = (1, 1, 1, 1), padding = 'VALID') + biases
        activations = tf.nn.relu(nets)

    return activations

def expand(input, channels_1by1, channels_3by3, layer_num):
    """
    Defines expand block for fire module.
    :param input: input tensor
    :param channels_1by1: number of output channels in 1x1 layers
    :param channels_3by3: number of output channels in 3x3 layers
    :param layer_num: layer number for naming purposes
    :return: output tensor convoluted with expand layer
    """

    layer_name = 'expand_' + layer_num
    input_channels = input.get_shape().as_list()[3]

    with tf.name_scope(layer_name):
        weights1x1 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, input_channels, channels_1by1]))
        biases1x1 = tf.Variable(tf.zeros([1, 1, 1, channels_1by1]), name = 'biases')
        nets_1x1 = tf.nn.conv2d(input, weights1x1, strides = (1, 1, 1, 1), padding = 'VALID') + biases1x1
        activations_1x1 = tf.nn.relu(nets_1x1)

        weights3x3 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, input_channels, channels_3by3]))
        biases3x3 = tf.Variable(tf.zeros([1, 1, 1, channels_3by3]), name = 'biases')
        nets_3x3 = tf.nn.conv2d(input, weights3x3, strides = (1, 1, 1, 1), padding = 'SAME') + biases3x3
        activations_3x3 = tf.nn.relu(nets_3x3)

    return tf.concat([activations_1x1, activations_3x3], axis = 3)

def fire_module(input, s_1x1_filters, e_1x1_filters, e_3x3_filters, layer_num):
    """
    Train fire module. Fire module does not change input height and width, only depth.
    :param input: input tensor
    :param s_1x1_filters: number of channels for 1x1 squeeze layer
    :param e_1x1_filters: number of channels for 1x1 expand layer
    :param e_3x3_filters: number of channels for 3x3 expand layer
    :param layer_num: number of layer for naming purposes only
    :return: a tensor of shape [input_height x input_width x expand_channels_1by1 * expand_channels_3by3]
    """
    with tf.name_scope('fire_' + layer_num):
        squeeze_output = squeeze(input, s_1x1_filters, layer_num)
        return expand(squeeze_output, e_1x1_filters, e_3x3_filters, layer_num)

def model(input_height, input_width, input_channels, output_classes, pooling_size = (1, 3, 3, 1)):
    """
    Define tensorflow graph.
    :param input_height: input image height
    :param input_width: input image width
    :param input_channels: input image channels
    :param output_classes: number of output classes
    :param pooling_size: size of the pooling
    :return: list of input placeholders and output operations
    """
    with tf.Graph().as_default() as graph:
        # define placeholders
        input_image = tf.placeholder(tf.float32, shape = [None, input_height, input_width, input_channels], name = 'input_image')
        labels = tf.placeholder(tf.int32, shape = [None, 1])
        in_training = tf.placeholder(tf.bool, shape = ())
        learning_rate = tf.placeholder(tf.float32, shape = ())

        # define structure of the net
        # layer 1 - conv 1
        with tf.name_scope('conv_1'):
            W_conv1 = tf.Variable(tf.contrib.layers.xavier_initializer()([7, 7, input_channels, 96]))
            b_conv1 = tf.Variable(tf.zeros([1, 1, 1, 96]))
            X_1 = tf.nn.conv2d(input_image, W_conv1, strides = (1, 2, 2, 1), padding = 'VALID') + b_conv1
            A_1 = tf.nn.relu(X_1)

        # layer 2 - maxpool
        maxpool_1 = tf.nn.max_pool(A_1, ksize = pooling_size, strides = (1, 2, 2, 1), padding = 'VALID', name = 'maxpool_1')

        # layer 3-5 - fire modules
        fire_2 = fire_module(maxpool_1, 16, 64, 64, "2")
        fire_3 = fire_module(fire_2, 16, 64, 64, "3")
        fire_4 = fire_module(fire_3, 32, 128, 128, "4")

        # layer 6 - maxpool
        maxpool_4 = tf.nn.max_pool(fire_4, ksize = pooling_size, strides = (1, 2, 2, 1), padding = 'VALID', name = 'maxpool_4')

        # layer 7-10 - fire modules
        fire_5 = fire_module(maxpool_4, 32, 128, 128, "5")
        fire_6 = fire_module(fire_5, 48, 192, 192, "6")
        fire_7 = fire_module(fire_6, 48, 192, 192, "7")
        fire_8 = fire_module(fire_7, 64, 256, 256, "8")

        # layer 11 - maxpool
        maxpool_8 = tf.nn.max_pool(fire_8, ksize = pooling_size, strides = (1, 2, 2, 1), padding = 'VALID', name = 'maxpool_8')

        # layer 12 - fire 9 + dropout
        fire_9 = fire_module(maxpool_8, 64, 256, 256, "9")

        dropout_9 = tf.cond(in_training, lambda: tf.nn.dropout(fire_9, keep_prob = 0.5), lambda: fire_9)

        # layer 13 - final
        with tf.name_scope('final'):
            W_conv10 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, 512, output_classes]))
            b_conv10 = tf.Variable(tf.zeros([1, 1, 1, output_classes]))
            conv_10 = tf.nn.conv2d(dropout_9, W_conv10, strides = (1, 1, 1, 1), padding = 'VALID') + b_conv10
            A_conv_10 = tf.nn.relu(conv_10)

        # avg pooling to get [1 x 1 x num_classes] must average over entire window oh H x W from input layer
        _, H_last, W_last, _ = A_conv_10.get_shape().as_list()
        pooled = tf.nn.avg_pool(A_conv_10, ksize = (1, H_last, W_last, 1), strides = (1, 1, 1, 1), padding = 'VALID')
        logits = tf.squeeze(pooled, axis = [1, 2])

        # loss + optimizer
        one_hot_labels = tf.one_hot(labels, output_classes, name = 'one_hot_encoding')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_hot_labels, logits = logits))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # accuracy
        predictions = tf.reshape(tf.argmax(tf.nn.softmax(logits), axis = 1, output_type = tf.int32), [-1, 1])
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype = tf.float32))

    return graph, input_image, labels, in_training, learning_rate, loss, accuracy, optimizer

def prepare_input(data, mean = None, standard_deviation = None):
    """
    Normalizes pixels across dataset. For training set, calculate mu and sigma. For test set, transfer these
    from training set.

    :param data: dataset
    :param mean: mean pixel value across dataset. Calculated if not provided.
    :param standard_deviation: standard deviation of pixel value across dataset. Calculated if not provided.
    :return: normalized dataset, mean and standard deviation
    """
    if mean is None:
        mean = np.mean(data)
    if standard_deviation is None:
        standard_deviation = np.std(data)
    data = data - mean
    data = data / standard_deviation
    return data, mean, standard_deviation

def load_dataset(dataset = "cifar10"):
    """
    Loads data set and returns information about dimensionality and train/test splits

    :param dataset: string value representing which data set to load
    :return: image_shape (h, w, d), num_classes, train_date (x, y), test_data (x, y)
    """
    if dataset == "cifar10":
        train, test = cifar10.load_data()
        return (32, 32, 3), 10, train, test
    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        y_train = np.expand_dims(y_train, -1)
        y_test = np.expand_dims(y_test, -1)
        return (28, 28, 1), 10, (x_train, y_train), (x_test, y_test)
    elif dataset == "imagenet":
        raise NotImplementedError("Haven't figured out ImageNet yet...")
        # return (227, 227, 3), 1500, ?, ?
    else:
        raise NotImplementedError("Unsupported data set: {}".format(dataset))

def run(iterations, minibatch_size):
    # Load dimensions and data
    (input_height, input_width, input_channels), output_classes, (x_train, y_train), (x_test, y_test) = load_dataset("mnist")

    x_train, mu_train, sigma_train = prepare_input(x_train)
    x_test, _, _ = prepare_input(x_test, mu_train, sigma_train)
    train_samples = x_train.shape[0]

    graph, input_batch, labels, in_training, learning_rate, loss, accuracy, optimizer = model(input_height, input_width, input_channels, output_classes, (1, 2, 2, 1))

    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            # pick random minibatch
            mb_start = np.random.randint(0, train_samples - minibatch_size)
            mb_end = mb_start + minibatch_size
            mb_data = x_train[mb_start:mb_end, :, :, :]
            mb_labels = y_train[mb_start:mb_end, :]

            _loss, _accuracy, _ = sess.run([loss, accuracy, optimizer], feed_dict = {
                input_batch: mb_data,
                labels: mb_labels,
                in_training: True,
                learning_rate: 0.0004
            })

            if i % 100 == 0:
                test_acc = sess.run(accuracy, feed_dict = {
                    input_batch: x_test,
                    labels: y_test,
                    in_training: False,
                    learning_rate: 0.0004
                })
                print('Iteration: {}\tloss: {:.3f}\t train accuracy: {:.3f}\ttest accuracy: {:.3f}'.format(i, _loss, _accuracy, test_acc))

run(10001, 128)
