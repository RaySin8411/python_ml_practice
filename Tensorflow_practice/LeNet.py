import os
import struct
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_mnist(path, kind='train'):
    """Load MNIST data from 'path'"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
    return images, labels


def batch_generator(X, y, batch_size=64, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size])


# Convolutional Layer
def conv_layer(input_tensor, name, kernel_size,
               n_output_channels, padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        ##get n_input_channels:
        ## input tensor shape:
        ## [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size) + [n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(shape=[n_output_channels]))
        conv = tf.nn.conv2d(input=input_tensor, filter=weights,
                            strides=strides, padding=padding_mode)
        conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
        conv = tf.nn.relu(conv, name='activation')
        return conv


# FC Layer
def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))
        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        biases = tf.get_variable(name='biases', initializer=tf.zeros(shape=[n_output_units]))
        layer = tf.matmul(input_tensor, weights)
        layer = tf.nn.bias_add(layer, biases, name='net_pre-activation')
        if activation_fn is None:
            return layer
        layer = activation_fn(layer, name='activation')
        return layer


def build_cnn(learning_rate: float):
    ## Placeholders for X and y:
    tf_x = tf.placeholder(tf.float32, shape=[None, 784], name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')
    # reshape to a 4D tensor:
    # [batchsize, width, height, 1]
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name='tf_x_reshaped')
    ## One-hot encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32)
    ## 1st layer: Conv_1
    h1 = conv_layer(tf_x_image, name='conv_1', kernel_size=(5, 5),
                    padding_mode='VALID', n_output_channels=32)
    ## MaxPooling
    h1_poo1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
    ## 2nd layer: Conv_2
    h2 = conv_layer(h1_poo1, name='conv_2', kernel_size=(5, 5),
                    padding_mode='VALID', n_output_channels=64)
    ## MaxPooling
    h2_poo1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    ## 3rd layer : Fully Connected
    h3 = fc_layer(h2_poo1, name='fc_3', n_output_units=1024,
                  activation_fn=tf.nn.relu)
    ## Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')
    ## 4th layer: Fully Connected (linear activation)
    h4 = fc_layer(h3_drop, name='fc_4', n_output_units=10, activation_fn=None)
    ## Prediction
    predictions = {'probabilities': tf.nn.softmax(h4, name='probabilities'),
                   'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')}
    ## Visualize the graph with TensorBoard
    ## Loss Function and Optimization
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')

    ## Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')
    ## Computing the prediction accuracy
    correct_predictions = tf.equal(predictions['labels'],
                                   tf_y, name='correct_preds')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                              name='accuracy')


def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step=epoch)


def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(path, 'cnn-model.ckpt-%d' % epoch))


def train(sess, training_set, validation_set=None, initialize=True,
          epochs=20, shuffle=True, dropout=0.5, random_seed=None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []
    ## initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())
    np.random.seed(random_seed)  # for shuffing in batch_generator
    for epoch in range(1, epochs + 1):
        batch_gen = batch_generator(X_data, y_data, shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y,
                    'fc_keep_prob:0': dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'],
                               feed_dict=feed)
        avg_loss += loss
        training_loss.append(avg_loss / (i + 1))
        print('Epoch %02d Training Avg Loss: %7.3f' % (epoch, avg_loss), end=' ')
        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0],
                    'tf_y:0': validation_set[1], 'fc_keep_prob:0': 1.0}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % valid_acc)


def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test, 'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
