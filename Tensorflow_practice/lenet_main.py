import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from Tensorflow_practice.LeNet import *


def main():
    ## Load Data
    X_data, y_data = load_mnist('C:/Users/USER/Desktop/structed_ml_application/code/', kind='train')
    print('Rows: %d, columns: %d ' % (X_data.shape[0], X_data.shape[1]))
    X_test, y_test = load_mnist('C:/Users/USER/Desktop/structed_ml_application/code/', kind='t10k')
    print('Rows: %d, columns: %d ' % (X_test.shape[0], X_test.shape[1]))
    X_train, y_train = X_data[:50000], y_data[:50000]
    X_valid, y_valid = X_data[50000:], y_data[50000:]
    print('Training: ', X_train.shape, y_train.shape)
    print('Validation: ', X_valid.shape, y_valid.shape)
    print('Test Set: ', X_test.shape, y_test.shape)

    # Standardization
    mean_vals = np.mean(X_train, axis=0)
    std_val = np.std(X_train)
    X_train_centered = (X_train - mean_vals) / std_val

    mean_vals = np.mean(X_valid, axis=0)
    std_val = np.std(X_valid)
    X_valid_centered = (X_valid - mean_vals) / std_val

    mean_vals = np.mean(X_test, axis=0)
    std_val = np.std(X_test)
    X_test_centered = (X_test - mean_vals) / std_val

    ## Define hyperparemeter
    learning_rate = 1e-4
    random_seed = 123
    ## create a graph
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(random_seed)
        ## bulid the graph
        build_cnn(learning_rate=learning_rate)
        ## saver:
        saver = tf.train.Saver()

    ## create a TF session
    ## and train the CNN model
    with tf.Session(graph=g) as sess:
        train(sess, training_set=(X_train_centered, y_train),
              validation_set=(X_valid_centered, y_valid), initialize=True, random_seed=random_seed)
        save(saver, sess, epoch=20)

    ### Calculate prediction accuracy on test set
    ### restoring the saved model
    del g
    ## create a new graph and build the model
    g2 = tf.Graph()
    with g2.as_default():
        tf.set_random_seed(random_seed)
        ## build the graph
        build_cnn()
        ## saver:
        saver = tf.train.Saver()
    ## create a new session
    ## and restore the model
    with tf.Session(graph=g2) as sess:
        load(saver, sess, epoch=20, path='./model/')
        preds = predict(sess, X_test_centered, return_proba=False)
        print('Test Accuracy: %.3f%%' % (100 * np.sum(preds == y_test) / len(y_test)))

    ## without re-initializing :: initialize=False
    ## create a new session
    ## and restore the model
    with tf.Session(graph=g2) as sess:
        load(saver, sess, epoch=20, path='./model/')
        train(sess, training_set=(X_train_centered, y_train), validation_set= \
            (X_valid_centered, y_valid), initialize=False, epochs=20, random_seed=123)
        save(saver, sess, epoch=40, path='./model/')
        preds = predict(sess, X_test_centered, return_proba=False)
        print('Test Accuracy: %.3f%%' % (100 * np.sum(preds == y_test) / len(y_test)))
