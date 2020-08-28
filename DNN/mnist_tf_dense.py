import tensorflow as tf
import os
from matplotlib import pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)
# The one_hot=True argument specifies that instead of the labels associated with each
# image being the digit itself i.e. "4", it is a vector with "one hot" node and all
# the other nodes being zero i.e. [0, 0, 0, 0, 1, 0, 0, 0, 0, 0].

# To suppress TF warnings instead the above do:
#mnist = tf.keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Optimisation variables
learning_rate = 0.0004  # 0.0005 for Adam, 0.5 for GD. Or use lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
epochs = 50  # 50 with dropout, 20 without
batch_size = 128
dropout = 0.75


# Declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784], name='input')

# Now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# For the dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


layer1size = 384
# Declare the weights connecting the input to the hidden layer (or first hidden layer weights)
# this neural network will have 512 nodes in the first hidden layer
W0 = tf.Variable(tf.truncated_normal([784, layer1size], stddev=0.1), name='W0')
b0 = tf.Variable(tf.ones([layer1size])/10, name='b0')  # For RELU. Otherwise random? Or const 1?

# Calculate the output of the first hidden layer, based on the input
hidden_out1 = tf.nn.relu(tf.add(tf.matmul(x, W0), b0))

# Add dropout here?


layer2size = 128
# Second hidden layer
W1 = tf.Variable(tf.truncated_normal([layer1size, layer2size], stddev=0.1), name='W1')
b1 = tf.Variable(tf.ones([layer2size])/10, name='b1')

# Calculate the output of the second hidden layer, based on the input
hidden_out2 = tf.nn.relu(tf.add(tf.matmul(hidden_out1, W1), b1))

# Add dropout. See https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow
layer_drop = tf.nn.dropout(hidden_out2, keep_prob)

# And the weights connecting the last hidden layer to the output layer
W2 = tf.Variable(tf.truncated_normal([layer2size, 10], stddev=0.1), name='W2')
b2 = tf.Variable(tf.ones([10])/10, name='b2')

# Output layer (predictions) - use a softmax activation
y_logits = tf.add(tf.matmul(layer_drop, W2), b2, name='output')
#y_logits = tf.add(tf.matmul(hidden_out2, W2), b2, name='output')
y_ = tf.nn.softmax(y_logits)


# Convert the output y_ to a clipped version, limited between 1e-10 to 0.999999.
# This is to make sure that we never get a case were we have a log(0) operation
# occurring during training – this would return NaN
#y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

# We also have to include a cost or loss function for the optimisation / backpropagation to work on.
# Here we’ll use the cross entropy cost function, represented by:
# J = −1/m ∑i=1..m ∑j=1..n y_j^(i) log(y_j_^(i)) + (1–y_j^(i)) log(1–y_j_^(i))
# Where y_j^(i) is the ith training label for output node j,
# y_j_^(i) is the ith predicted label for output node j, m is the number of training / batch samples
# and n is the number of digits.
# Remember that y and y_clipped in the above calculation are (m x 10) tensors – therefore we need
# to perform the first sum over the second axis.  This is specified using the axis=1 argument, where
# "1" actually refers to the second axis when we have a zero-based indices system like Python.
# After this operation, we have an (m x 1) tensor.  To take the mean of this tensor and complete our
# cross entropy cost calculation (i.e. execute this part 1/m∑i=1..m), we use TensorFlow's tf.reduce_mean.
#cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
#                                            + (1 - y) * tf.log(1 - y_clipped), axis=1))

# TensorFlow has a handy function that computes the softmax and the cross-entropy in a single step,
# implemented in a numerically stable way.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_logits, labels=y))


# Add an optimiser. Can be tf.train.GradientDescentOptimizer.
# In very high dimensional spaces like here - we have in the order of 10K weights and biases -
# "saddle points" are frequent. These are points that are not local minima but where the gradient
# is nevertheless zero and the gradient descent optimizer stays stuck there. Adam is better.
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


# Finally setup the initialisation operator
init_op = tf.global_variables_initializer()


# Create a saver object which will save all the variables
saver = tf.train.Saver()


# Define an accuracy assessment operation.
# The tf.argmax function is the same as the numpy argmax function, which returns
# the index of the maximum value in a vector / tensor.  Therefore, the
# correct_prediction operation returns a tensor of size (m x 1) of True and False
# values designating whether the neural network has correctly predicted the digit.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # returns a tensor of size (m x 1) of True and False
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Start the training session
with tf.Session() as sess:
    imported_graph = None
    if os.path.isfile(os.path.abspath("mnist_tf_dense_model.meta")):
        imported_graph = tf.train.import_meta_graph(os.path.abspath('mnist_tf_dense_model.meta'))  # Don't need this probably

    if imported_graph is not None:
        print("Loading saved model..")
        imported_graph.restore(sess, os.path.abspath("mnist_tf_dense_model"))  # Or use the saver
    else:
        print("Initializing variables...")
        sess.run(init_op)  # Initialise the variables

    test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
    print("Before training! Test accuracy = ", test_acc)

    if test_acc < 0.98 :  # Only train if not trained already
        total_batch = int(len(mnist.train.labels) / batch_size)
        losses = []
        for epoch in range(epochs):  # Or require some accuracy to stop
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, ce = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                avg_cost += ce
            avg_cost /= total_batch
            print("Epoch:", (epoch + 1), "loss =", "{:.4f}".format(avg_cost))
            losses.append(avg_cost)
            if avg_cost < 0.001: break

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Training finished! Test accuracy = ", test_acc)

        if len(losses) > 1:
            plt.plot(losses)
            plt.xlabel("Epochs")
            plt.ylabel("Training loss")
            plt.show()

        # Save the model
        saver.save(sess, os.path.abspath("mnist_tf_dense_model"))

# Some tests
#    if os.path.isfile(os.path.abspath("digit.png")):
#        img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
#        plt.imshow(img, cmap="Greys")
#        plt.show()
#        newimg = (img.astype(float) / 255.).reshape(784)
#        predictions = sess.run(y_, feed_dict={x: [newimg], keep_prob: 1.0})
#        print(predictions, np.argmax(predictions))
#
#    for i in range(5,15) :
#        mimg = mnist.test.images[i]
#        plt.imshow(mimg.reshape(28,28), cmap="Greys")
#        plt.show()
#        predictions = sess.run(y_, feed_dict={x: [mimg], keep_prob: 1.0})
#        print(predictions, np.argmax(predictions), mimg)


import cv2
import numpy as np
import sys
sys.path.append('../Vision/OpenCV/')
import cv_draw as cvd


# Reset TF graph and vars
tf.reset_default_graph()

# Import saved graph
imported_graph = tf.train.import_meta_graph(os.path.abspath('mnist_tf_dense_model.meta'))

# Run recognition session
with tf.Session() as sess:
    # Restore saved variables
    imported_graph.restore(sess, os.path.abspath("mnist_tf_dense_model"))
    output_tensor = sess.graph.get_tensor_by_name('output:0')

    while True:
        # Draw with the mouse
        img = cvd.draw(384, 384, 'Draw a digit and press Enter to recognize. Esc to close')
        if img is None:
            break
        else:
            plt.imshow(img, cmap="Greys")
            plt.show()
            newimg = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            #cv2.imwrite("digit.png", newimg)
            newimg = (newimg.astype(float) / 255.).reshape(784)
            # Make a prediction
            predictions = sess.run(tf.nn.softmax(output_tensor), {'input:0': [newimg], 'keep_prob:0': 1.0})
            print(predictions, np.argmax(predictions))
