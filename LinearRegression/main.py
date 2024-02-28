import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model

import graphviz
import os

def main():
    # Load the data
    df = pd.read_csv('C:\\Users\\PC\\Downloads\\ML\\Machine Learning Practise\\LinearRegression\\height.csv')

    height = df['Height']
    weight = df['Weight']


    # Create the model
    plt.plot(height, weight, 'ro')
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()

    # Create the model
    X = height.values.reshape(-1, 1)
    y = weight.values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    
    # Print the model
    print('Intercept:', model.intercept_)
    print('Coefficient:', model.coef_)
    print('R^2:', model.score(X_test, y_test))

    # Plot the model
    plt.plot(X, y, 'ro')
    plt.plot(X, model.predict(X), color='blue')
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()

    # Predict
    height = 170
    weight = model.predict([[height]])
    print('Weight:', weight)


    # Create the model
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    # Print the model
    print('Intercept:', model.intercept_)
    print('Coefficient:', model.coef_)
    print('R^2:', model.score(X_test, y_test))

    # Plot the model
    plt.plot(X, y, 'ro')
    plt.plot(X, model.predict(X), color='blue')
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()

 
 

if __name__ == '__main__':
    main()
