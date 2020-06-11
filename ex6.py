import numpy as np
from itertools import product
from random import gauss
import math

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
    return biased_truth_table[:,0:4].T,biased_truth_table[4]

def initialize_hidden_weights_matrix():
    """
    Initializes the weights matrix from samples of gaussian distribution
    with 0 mean and 1 variance.
    Every row represents the weights for the z_i neuron.
    :return: w(3X4): the weights matrix
    """
    w=[[gauss(0,1) for i in range(4)] for j in range(3)]
    return np.array(w)

def initialize_output_weights_vector():
    w=[gauss(0,1) for i in range(4)]
    return np.array(w)

def calculate_nodes(hidden_w,output_w,x):
    """

    :param hidden_w: the weights matrix for the hidden layer. 3X4
    :param output_w: the weights vector for the output node. 1X4
    :param x: the biased input matrix. 4X8
    :return: Y - vector of outputs for every input vector. 1X8
    """
    hidden_arguments_matrix=hidden_w.dot(x)
    Z=sigmoid(hidden_arguments_matrix) #z is now 3X8 matrix, the i-th row is Z_i
    biased_Z=np.r_[np.reshape(np.ones(8),(1,8)),Z] #biased _Z is 4X8 matrix, the first row is ones for bias
    output_arguments_vector=output_w.dot(biased_Z) #output_arguments_vector is now 1X8, where the i-th value corresponds the i-th input vector.
    Y=sigmoid(output_arguments_vector) #passing the output arguments through the sigmoid
    return Y #returning the output vector (1X8, one output for every input vector)

def sigmoid(m):
  return 1 / (1 + np.exp(-m))

if (__name__=="__main__"):
    x,t=initialize_truth_table()
    print(x)
    print("PlaceHolder")

