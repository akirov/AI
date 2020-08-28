Deep Neural Networks
====================

Handwritten digit recognition
-----------------------------

Several Python implementations:

**mnist_tf_dense.py** - Tensorflow implementation using one 384-neuron hidden
dense layer with relu activation, a 128-neuron hidden dense layer with dropout
for regularization, and a softmax output layer.

**mnist_tf_cnn.py** - Tensorflow implementation using a convolutional neural
network with several convolution layers, a fully connected layer, an output
layer and a variable learning step.

After the training, the last two scripts open a window where you can draw a
digit with the mouse and it will be recognized. OpenCV is used for drawing.
