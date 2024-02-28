
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import idx2numpy
from mnist import MNIST

import graphviz
import os

def main():
    # Load the data
    mndata = MNIST('C:/Users/PC/Downloads/ML/Machine Learning Practise/KMean/')

    train_image, train_label = mndata.load_training()
    # or
    test_image, test_label = mndata.load_testing()

    # Reshape the data
    train_image = np.array(train_image)
    train_label = np.array(train_label)
    test_image = np.array(test_image)
    test_label = np.array(test_label)

    # Create the model
    K = 10
    model = KMeans(n_clusters= K)
    
    # Train the model
    model.fit(train_image)
    
    # Predict the model
    test_predict = model.predict(test_image)
    train_predict = model.predict(train_image)

    # Evaluate the model
    test_accuracy = np.mean(test_predict == test_label)
    train_accuracy = np.mean(train_predict == train_label)
    print('Test accuracy: ', test_accuracy)
    print('Train accuracy: ', train_accuracy)

    
    
if __name__ == '__main__':
    main()
