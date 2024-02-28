import pandas as pd


def train_test_split(X, y, train_percent, test_percent):
    
    X_train, X_test = split_data_percent(X, train_percent, test_percent)
    y_train, y_test = split_data_percent(y, train_percent, test_percent)
    

    return X_train, X_test, y_train, y_test

    
    
def split_data_percent(data, train_percent, test_percent):
    # Calculate the actual percentages
    test_percent = test_percent / (1 - train_percent)
    
    # Split into train and temp
    train_data, test_data = split_data(data, test_size=1-train_percent)
    
    return train_data, test_data

def split_data(data: pd.DataFrame, test_size):
    train_data = data.iloc[:int(len(data) * (1 - test_size))]
    test_data = data.iloc[int(len(data) * (1 - test_size)):]
    
    return train_data, test_data
    
