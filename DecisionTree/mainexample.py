import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing



import graphviz
import os

def main():
    # Load the data
    df = pd.read_csv('drug200.csv')

    sexs = ['F', 'M']
    blood_pressure = ['LOW', 'NORMAL', 'HIGH']
    cholesterols = ['NORMAL', 'HIGH']
    enc = preprocessing.OneHotEncoder(categories=[sexs, blood_pressure, cholesterols])
 
    # Split the DataFrame into features and target
    X = df.drop('Drug', axis=1)
    y = df['Drug']

    # Apply the OneHotEncoder to the categorical columns
    X_encoded = enc.fit_transform(X[['Sex', 'BP', 'Cholesterol']]).toarray()

    # Create a DataFrame from the encoded data, with column names
    X_encoded_df = pd.DataFrame(X_encoded, columns=enc.get_feature_names_out(['Sex', 'BP', 'Cholesterol']))

    # Replace the old columns with the new one-hot encoded columns
    X = pd.concat([X.drop(['Sex', 'BP', 'Cholesterol'], axis=1), X_encoded_df], axis=1)

    #iris = datasets.load_iris()
    #X = iris.data
    #y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=43, shuffle=True)

    # Create the model
    model = tree.DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    print('Accuracy: {}'.format(model.score(X_test, y_test)))

    # Create a visualization of the tree
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=X.columns, class_names=y.unique())
    graph = graphviz.Source(dot_data)
    graph.render('drug')

    os.system("start drug.pdf")


if __name__ == '__main__':
    main()
