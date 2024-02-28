import pandas as pd
import numpy as np

import DecisionTreePacket.DecisionTree as DecisionTree
import Process.ModelSelection as ms

def main():

    file = 'C:/Users/PC/Downloads/ML/Machine Learning Practise/DecisionTree/golf.csv'
    df = pd.read_csv(file)

    X = df.drop('PlayGolf', axis=1)
    y = df['PlayGolf']

    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_percent=0.8, test_percent=0.2)

    # Create the model
    model = DecisionTree.DecisionTree()
    model.fit(X_train, y_train)

    
    print('Accuracy: {}'.format(model.score(X_test, y_test)))


    
    pass



if __name__ == '__main__':
    main()    
