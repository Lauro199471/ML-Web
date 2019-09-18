import numpy as np 
from helper import *

'''
Homework1: perceptron classifier
'''
def sign(x):
    return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):  
    '''''
    This function is used for plot image and save it.
    Args:
    data: Two images from train data with shape (2, 16, 16). The shape resents total 2 images and each image has size 16 by 16.
    Returns:
    Do not return any arguments, just save the images you plot for your report.
    '''''
    for iImag in range(len(data)):
        dataRow = data[iImag][1:]
        pixels  = np.matrix(dataRow)   
        plt.figure(figsize=(2.5,2.5))
        plt.imshow(pixels ,cmap='gray')
        plt.show() 


def show_features(data, label):
    '''
    This function is used for plot a 2-D scatter plot of the features and save
    it.  
    Args:
    data: train features with shape (1561, 2). The shape represents total 1561 
    samples and each sample has 2 features.
    
    label: train data's label with shape (1561, 1).
      1 for digit number 1  
     -1 for digit number 5. 
    Returns:
    Do not return any arguments, just save the 2-D scatter plot of the features
    you plot for your report.
    '''
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data[:,0][label == 1], data[:,1][label == 1] , color='red', marker='*')
    ax.scatter(data[:,0][label == -1], data[:,1][label == -1],color='blue', marker='+')    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set(xlim=(-0.95, .01), ylim=(-0.9, 0.15), xlabel='Symmetry', 
    ylabel='Average Intensity', title='Handwritten Digits Regconition')
    ax.set_facecolor('#f0f0f0')



def perceptron(data, label, max_iter, learning_rate):
    '''
    The perceptron classifier function.

    Args:
    data: train data with shape (1561, 3), which means 1561 samples and 
          each sample has 3 features.(1, symmetry, average internsity)
    label: train data's label with shape (1561,1). 
           1 for digit number 1 and -1 for digit number 5.
    max_iter: max iteration numbers
    learning_rate: learning rate for weight update
    
    Returns:
        w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
    '''
    n,m = data.shape
    w = np.zeros((1,m))
    
    for t in range(max_iter):
        for i, x in enumerate(data):
            prediction = 1.0 if np.dot(w,data[i]) >= 0.0 else -1.0
            if(prediction != label[i]):
                w = w + (learning_rate * label[i] * data[i])
    return w



def show_result(data, label, w):
    '''
    This function is used for plot the test data with the separators and save it.
    
    Args:
    data: test features with shape (424, 2). The shape represents total 424 samples and 
          each sample has 2 features.
    label: test data's label with shape (424,1). 
           1 for digit number 1 and -1 for digit number 5.
    
    Returns:
    Do not return any arguments, just save the image you plot for your report.
    '''
    b  = w[0,0]
    wx = w[0,1]
    wy = w[0,2]
    x = np.linspace(-0.4,-0.2,100)
    y = (-b-(x*wx))/wy
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, '-g')
    ax.scatter(data[:,0][label == 1], data[:,1][label == 1] , color='red',
    marker='*')
    ax.scatter(data[:,0][label == -1], data[:,1][label == -1],color='blue',
    marker='+')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set(xlim=(-0.95, .01), ylim=(-0.9, 0.15), xlabel='Symmetry', 
    ylabel='Average Intensity', title='Handwritten Digits Regconition')
    ax.set_facecolor('#f0f0f0')



#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
    n, _ = data.shape
    mistakes = 0
    for i in range(n):
        if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
            mistakes += 1
    return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
    #get data
    traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
    train_data,train_label = load_features(traindataloc)
    test_data, test_label = load_features(testdataloc)
    #train perceptron
    w = perceptron(train_data, train_label, max_iter, learning_rate)
    train_acc = accuracy_perceptron(train_data, train_label, w)    
    #test perceptron model
    test_acc = accuracy_perceptron(test_data, test_label, w)
    return w, train_acc, test_acc


