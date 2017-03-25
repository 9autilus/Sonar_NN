import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.my_model import create_model, train_model, test_model
from sklearn.metrics import accuracy_score as acc

n_folds = 5 #13
n_epochs = 10

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_file = dir_path + r'\data\sonar.all-data.csv'
    x_raw = np.genfromtxt(data_file, delimiter=',', dtype='str')
    X = x_raw[:, :x_raw.shape[1]-1].astype('float')
    y = (x_raw[:, x_raw.shape[1]-1] == 'R').astype(int)

    input_shape = (X.shape[1],)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    skf.get_n_splits(X, y)


    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print('\rRunning Fold', i + 1, '/', n_folds)
        model = None  # Clearing the NN.
        model = create_model(input_shape)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = train_model(model, X_train, y_train, n_epochs)
        y_score = test_model(model, X_test)
        y_pred = y_score > 0.5

        accuracy = acc(y_test, y_pred)
        print('Accuracy: ', accuracy)

    exit(0)