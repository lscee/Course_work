import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    hypothesis = 0.0
    for index in range(len(theta)):
        hypothesis = hypothesis + theta[index] * X [i,index]
        #print(X [i,index])
    ########################################/
    
    return hypothesis
