import numpy as np 
from helper import *

'''
Homework2: logistic regression classifier
'''
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def logistic_regression(data, label, max_iter, learning_rate):
    '''
    The logistic regression classifier function.
    
    Args:
    data: train data with shape (1561, 3), which means 1561 samples and 
      each sample has 3 features.(1, symmetry, average internsity)
      
    label: train data's label with shape (1561,1). 
       1 for digit number 1 and -1 for digit number 5.
       
    max_iter: max iteration numbers
    
    learning_rate: learning rate for weight update
    
    Returns:
    w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
    '''
    N,m = data.shape
    w = np.zeros((m,1))
    # Compute the Gradient
    for t in range(max_iter):
        g = 0
        for n in range(N):
            g = (-label[n]*data[n]*sigmoid(-label[n]*np.dot(w.T,data[n]))) + g
        g = (1/N)*g
        g = g.reshape((m,1))
        w = w + (learning_rate * -g)    
    return w


def thirdorder(data):
    '''
    This function is used for a 3rd order polynomial transform of the data.
    Args:
    data: input data with shape (:, 3) the first dimension represents 
          total samples (training: 1561; testing: 424) and the 
          second dimesion represents total features.

    Return:
    result: A numpy array format new data with shape (:,10), which using 
            a 3rd order polynomial transformation to extend the feature numbers 
            from 3 to 10. 
            The first dimension represents total samples (training: 1561; testing: 424) 
            and the second dimesion represents total features.
    '''
    N,_ = data.shape
    ones = np.ones((N,1))
    x_one_pow_two = np.power(data[:,0:1],2)
    x_two_pow_two = np.power(data[:,1:2],2)
    x_one_pow_thr = np.power(data[:,0:1],3)
    x_two_pow_thr = np.power(data[:,1:2],3)
    
    result = np.hstack((ones,                      #  1
                        data[:,0:1],               # x_1
                        data[:,1:2],               # x_2
                        x_one_pow_two,             # (x_1)^2
                        data[:,0:1]*data[:,1:2],   # x_1*x_2
                        x_two_pow_two,             # (x_2)^2
                        x_one_pow_thr,             # (x_1)^3
                        x_one_pow_two*data[:,1:2], # (x_1)^2 * x_2
                        data[:,0:1]*x_two_pow_two, # x_1 * (x_2)^2
                        x_two_pow_thr              # (x_2)^3
                       ))       

    return result


def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    final_scores = np.dot(x, w)
    preds = sigmoid(final_scores)
    preds = [1 if(x >= 0.5) else -1 for x in preds]
    top = 0
    for i in range(len(y)):
        if(preds[i]==y[i]):
            top = top + 1
    accuracy = top/len(y)
    return accuracy

