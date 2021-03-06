import numpy as np
from itertools import product
from random import gauss
import matplotlib.pyplot as plt

"""
Exercise 6 in ML course, TAU
Authors:
Ohad Nir <302342589>
Valentin Krasny <209224369>
---------------------------
Parity-3 calculation by FF neural network
with 1 hidden layer, contains K nodes.
activation function: sigmoid
----------------------------
MSE typical results:
K=3: 0.00895
K=6: 0.00020
K=15: 0.00015295
K=30: 0.0001276
----------------------------
For further documantation, see README.pdf on: https://github.com/OhadNir9/ML6
"""


K=3

def initialize_truth_table():
    """
    Initializes the truth table and biases it.
    This will be the training data.
    :return:
    x(4X8): the 8 possible input vectors (columns), where the bias term is the first one.
    t(8X1): the tags vector.
    """
    truth_table=list(product((0,1),repeat=3))
    truth_table = [list(elem) for elem in truth_table]
    for i in range(len(truth_table)):
        truth_table[i].append(truth_table[i][0]^truth_table[i][1]^truth_table[i][2])
    truth_table=np.array(truth_table)
    biased_truth_table=np.c_[np.ones(8),truth_table]
    return biased_truth_table[:,0:4].T,biased_truth_table[:,4].reshape((8,1))

def initialize_hidden_weights_matrix():
    """
    Initializes the weights matrix from samples of gaussian distribution
    with 0 mean and 1 variance.
    Every row represents the weights for the z_i neuron.
    :return: w(KX4): the weights matrix
    """
    w=[[gauss(0,1) for i in range(4)] for j in range(K)]
    return np.array(w)

def initialize_output_weights_vector():
    """
    Initializes the weights vector for the output node (hidden to output), including bias weight
    :return: w(1XK+1)
    """
    w=[gauss(0,1) for i in range(K+1)]
    return np.array(w).reshape((1,K+1))

def calculate_nodes(hidden_w,output_w,x):
    """

    :param hidden_w: the weights matrix for the hidden layer. KX4
    :param output_w: the weights vector for the output node. 1X(K+1)
    :param x: the biased input matrix. 4X8
    :return: biased_Z - matrix of the node values (biased) for every input. (K+1)X8.
    :return: Y - vector of outputs for every input vector. 1X8
    """
    hidden_arguments_matrix=hidden_w.dot(x) #(KX4)dot(4X8)=(KX8)
    Z=sigmoid(hidden_arguments_matrix) #z is now KX8 matrix, the i-th row is Z_i
    biased_Z=np.r_[np.reshape(np.ones(8),(1,8)),Z] #biased _Z is (K+1X8) matrix, the first row is ones for bias
    output_arguments_vector=output_w.dot(biased_Z) #output_arguments_vector is now 1X8, where the i-th value corresponds the i-th input vector.
    Y=sigmoid(output_arguments_vector) #passing the output arguments through the sigmoid
    return biased_Z,Y #returning the biased Z matrix(K+1X8) output vector (1X8, one output for every input vector)

def calculate_loss(y,t):
    """
    calculates the mean sqaure error loss
    :param y: outputs vector 1X8
    :param t: the tags vector 8X1
    :return: the mse, scalar
    """
    return ((y-t.T)**2).mean(axis=None)

def sigmoid(m):
  return 1 / (1 + np.exp(-m))


def gradient_descent(X_biased,hidden_w_biased,output_w_biased,Z_biased,y,t):
    """
    Makes gradient descent step using backpropagation method.
    returns the updated weights (hidden_w matrix and output_w vector) including biases.
    :param X_biased: 4X8
    :param hidden_w_biased: KX4
    :param output_w_biased: 1X(K+1)
    :param Z_biased: (K+1)X8
    :param y: 1X8
    :param t: 8X1
    :return: new_hidden_w_biased (KX4), new_output_w_biased(1XK+1)
    """
    delta_out=y*(1-y)*(y-t.T) #delta out should be 1X8 (element wise multiplication between 1X8 arrays)
    output_w_unbiased=output_w_biased[:,1:] #Now (1XK)
    Z_unbiased=Z_biased[1:,:] #Now (KX8)
    delta_hidden=(output_w_unbiased.T.dot(delta_out))*(Z_unbiased*(1-Z_unbiased)) #element wise multiplication of KX8 arrays.
    new_output_w_biased=output_w_biased-2*(delta_out.dot(Z_biased.T)) # 1XK+1. the dot product is (1X8)dot(8XK+1)
    new_hidden_w_biased=hidden_w_biased-2*(delta_hidden.dot(X_biased.T)) #KX4. the dot product is (KX8)dot(8X4)
    return new_hidden_w_biased,new_output_w_biased


if (__name__=="__main__"):
    X_biased,t=initialize_truth_table()
    losses=np.zeros((100,2000))

    for i in range(100):
        hidden_w_biased=initialize_hidden_weights_matrix()
        output_w_biased=initialize_output_weights_vector()
        Z_biased,Y=calculate_nodes(hidden_w_biased,output_w_biased,X_biased)
        for j in range(2000):
            loss=calculate_loss(Y,t)
            losses[i,j]=loss
            hidden_w_biased,output_w_biased=gradient_descent(X_biased,hidden_w_biased,output_w_biased,Z_biased,Y,t)
            Z_biased,Y=calculate_nodes(hidden_w_biased,output_w_biased,X_biased)

    plt.plot(list(range(2000)),losses.mean(axis=0))
    print("Final avarage min loss: "+str(losses.mean(axis=0)[1999]))
    print("Weights from the last iteration, last run: ")
    for i in range(K):
        print("Z{} weights: ".format(i+1))
        print(hidden_w_biased[i])
    print("output weights: ")
    print(output_w_biased.reshape(1,K+1))
    plt.xlabel("Iteration num.")
    plt.ylabel("Mean square error")
    plt.title("MSE Vs. Iteration number \n{}-node-hidden-layer \navarage over 100 runs with 2000 iterations each".format(K))
    plt.show()
