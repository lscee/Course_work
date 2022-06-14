import numpy as np
from sigmoid import *

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis = 0.0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    #print(len(X[i]))
    #print(X[1,0])
    #print(theta)
    #print(len(theta))
    hypothesis=np.matmul(X[i],theta.T)
    #print("hypothesis:",hypothesis)
    ########################################/
    result = sigmoid(hypothesis)
    
    return result
X=[[1,10,20],[1,20,30]]
theta=[0.5,0.6,0.7]
#print(calculate_hypothesis(X,theta,0))