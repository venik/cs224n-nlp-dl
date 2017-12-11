#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    # print('=> ss ' + str(params))
    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    H_sigmoid = sigmoid(data.dot(W1) + b1)
    Y_softmax = softmax(H_sigmoid.dot(W2) + b2)
    cost = -1 * np.sum(labels * np.log(Y_softmax))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    #raise NotImplementedError
    # delta_1 = Y_softmax - labels
    # # print('delta_1: ' + str(delta_1.shape))
    # gradW2 = np.dot(H_sigmoid.T, delta_1)
    # gradb2 = np.sum(delta_1, axis=0)

    # delta_2 = gradW2 = delta_1.dot(W2.T)
    # delta_3 = delta_2 * sigmoid_grad(H_sigmoid)
    # # delta_3 = delta_2.T.dot(sigmoid_grad(H_sigmoid))

    # gradW1 = np.dot(data.T, delta_3)
    # gradb1 = np.sum(delta_3, axis=0)

    dl2 = Y_softmax - labels
    gradW2 = np.dot(H_sigmoid.T, dl2)
    gradb2 = np.sum(dl2, axis=0)

    dh = np.dot(dl2, W2.T)

    dl1 = dh * sigmoid_grad(H_sigmoid)
    gradW1 = np.dot(data.T, dl1)
    gradb1 = np.sum(dl1, axis=0)


    # print('=========')
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    # print('cost: ' + str(cost) + ' grad: ' + str(grad))
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
