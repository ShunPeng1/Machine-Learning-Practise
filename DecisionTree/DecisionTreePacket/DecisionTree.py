import pandas as pd
import numpy as np


class DecisionTree:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        
        
        
        pass

    def score(self, X, y):
        pass

    def predict(self, X):
        pass

    def get_gain(self, X, y, feature):
        pass

    def get_class_values(self, X, y, feature, value):
        values = []



        for i in range(len(X)):
            if X[i][feature] == value:
                values.append(y[i])
        
                # Check the type of the first value in the list
        if values:
            first_value = values[0]
            if isinstance(first_value, str):
                print("The data is a string.")
            elif isinstance(first_value, (int, float)):
                print("The data is a number.")
            else:
                print("The data is neither a string nor a number.")
            
        return values
        

    def information_gain(self, y, y_left, y_right):

        pass

    def entropy(self, values : list):
        total = sum(values)
        entropy = 0

        for i in range(len(values)):
            proportion = self.proportion(values[i], total)
            entropy += -proportion*numpy.log2(proportion)
            
        return entropy

    def proportion(self, p, n):
        return p/(p+n)    

    def information_gain(self, y, y_left, y_right):
        pass

    def gini(self, y):
        pass
    


