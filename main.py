from __future__ import print_function
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.my_model import create_model, train_model, test_model
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

seed = 1337  # for reproducibility
import numpy as np
np.random.seed(seed)        # Seed Numpy
import random               # Seed random
random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)    # Seed Tensor Flow

n_folds = 5
n_epochs = 150
model_name = 'model.h5'


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def train_and_test(X, y, skf, n_epochs, n_folds, num_layers, num_units, other_args):
    input_shape = (X.shape[1],)
    loss = np.empty([n_folds, n_epochs])
    acc = np.empty([n_folds, n_epochs])
    val_acc = np.empty([n_folds, n_epochs])
    val_loss = np.empty([n_folds, n_epochs])

    conf_matrices = np.empty([n_folds, 2, 2])

    # Loop over all k-folds
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print('Running Fold {0:d}/{1:d}'.format(i, n_folds))
        model = None  # Clearing the NN.
        model = create_model(input_shape, num_layers, num_units)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        history_fold = train_model(model, model_name, X_train, y_train, X_test, y_test, n_epochs, other_args)

        loss[i] = history_fold.history['loss']
        acc[i] = history_fold.history['acc']
        val_loss[i] = history_fold.history['val_loss']
        val_acc[i] = history_fold.history['val_acc']

        y_score = test_model(model_name, X_test)
        y_pred = y_score > 0.5 # Thresholding

        # Compute confusion matrix
        conf_matrices[i] = confusion_matrix(y_test, y_pred)

    # Average-out performance data over all k-folds
    history = {}
    history['loss'] = np.mean(loss, axis=0)
    history['acc'] = np.mean(acc, axis=0)
    history['val_loss'] = np.mean(val_loss, axis=0)
    history['val_acc'] = np.mean(val_acc, axis=0)
    conf_matrix = np.mean(conf_matrices, axis=0)

    return history, conf_matrix


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_file = dir_path + r'\data\sonar.all-data.csv'
    x_raw = np.genfromtxt(data_file, delimiter=',', dtype='str')
    X = x_raw[:, :x_raw.shape[1]-1].astype('float')
    y = (x_raw[:, x_raw.shape[1]-1] == 'R').astype(int)

    # Bringing data to range (-0.5, 0.5)
    X = X - 0.5


    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    skf.get_n_splits(X, y)

    exp_00 = 0
    exp_01 = 0
    exp_02 = 0
    exp_03 = 1

    # Accuracy vs epochs vs #layers
    if exp_00:
        exp_file_name = 'exp_00.npy'
        layers = [1,3, 5,7, 10]
        units = 128
        other_args = {}

        if 0: # generate data
            all_history = []
            all_conf_matrix = []

            for i in layers:
                print('#### Num layers: {0:d}'.format(i))
                num_layers = i
                num_units = units
                history, conf_matrix = train_and_test(X, y, skf, n_epochs, n_folds, num_layers, num_units, other_args)
                all_history.append(history)
                all_conf_matrix.append(conf_matrix)
            exp_data = {'history': all_history, 'conf_matrix': all_conf_matrix}
            np.save(exp_file_name, exp_data)
        else: # Read data
            exp_data = np.load(exp_file_name).item()
            lw = 1

            if 1: # Training set: Accuracy vs Epochs vs #Layers
                plt.figure()
                plt.grid()
                plt.title('Training-Set: Accuracy vs Epochs vs #Hidden-Layers')  # summarize history for accuracy
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                legend = []
                for i in range(len(layers) - 1):  # -1 to ignore the last weird result
                    data = exp_data['history'][i]['acc']
                    data = smooth(data, window_len=11)
                    plt.plot(data, linewidth=lw)
                    legend.append('#Hidden-Layers: ' + str(layers[i]))
                plt.legend(legend, loc='lower right')
                plt.show()

            if 1: # Test set: Accuracy vs Epochs vs #Layers
                plt.figure()
                plt.grid()
                plt.title('Test-Set: Accuracy vs Epochs vs #Hidden-Layers')  # summarize history for accuracy
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                legend = []
                for i in range(len(layers) - 1):  # -1 to ignore the last weird result
                    data = exp_data['history'][i]['val_acc']
                    data = smooth(data, window_len=11)
                    plt.plot(data, linewidth=lw)
                    legend.append('#Hidden-Layers: ' + str(layers[i]))
                plt.legend(legend, loc='lower right')
                plt.show()

            if 1: # Loss vs epochs vs phase
                plt.figure()
                plt.grid()
                plt.title('Loss vs Epochs vs #Hidden-Layers')  # summarize history for accuracy
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                legend = ['Training Loss', 'Testing Loss']
                best_layer_id = 1
                # Training loss
                data = exp_data['history'][best_layer_id]['loss']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                # Validation loss
                data = exp_data['history'][best_layer_id]['val_loss']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                plt.legend(legend, loc='best')
                plt.show()

            if 1: # Accuracy vs epochs vs phase
                plt.figure()
                plt.grid()
                plt.title('Accuracy vs Epochs vs #Hidden-Layers')  # summarize history for accuracy
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                legend = ['Training Accuracy', 'Testing Accuracy']
                best_layer_id = 1
                # Training acc
                data = exp_data['history'][best_layer_id]['acc']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                # Validation acc
                data = exp_data['history'][best_layer_id]['val_acc']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                plt.legend(legend, loc='best')
                plt.show()

    # Accuracy vs epochs vs #Units
    if exp_01:
        exp_file_name = 'exp_01.npy'
        layers = 3
        units = [8, 16, 32, 64, 128]
        other_args = {}

        if 0: # generate data
            all_history = []
            all_conf_matrix = []

            for i, unit in enumerate(units):
                print('#### Num Units: {0:d}'.format(unit))
                num_layers = layers
                num_units = unit
                history, conf_matrix = train_and_test(X, y, skf, n_epochs, n_folds, num_layers, num_units, other_args)
                all_history.append(history)
                all_conf_matrix.append(conf_matrix)
            exp_data = {'history': all_history, 'conf_matrix': all_conf_matrix}
            np.save(exp_file_name, exp_data)
        else: # Read data
            exp_data = np.load(exp_file_name).item()
            lw = 1

            if 1:  # Training set: Accuracy vs Epochs vs #Units
                plt.figure()
                plt.grid()
                plt.title('Training-Set: Accuracy vs Epochs vs #Units')  # summarize history for accuracy
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                legend = []
                for i in range(len(units)):
                    data = exp_data['history'][i]['acc']
                    data = smooth(data, window_len=11)
                    plt.plot(data, linewidth=lw)
                    legend.append('#Units: ' + str(units[i]))
                plt.legend(legend, loc='lower right')
                plt.show()

            if 1:  # Test set: Accuracy vs Epochs vs #Units
                plt.figure()
                plt.grid()
                plt.title('Test-Set: Accuracy vs Epochs vs #Units')  # summarize history for accuracy
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                legend = []
                for i in range(len(units)):
                    data = exp_data['history'][i]['val_acc']
                    data = smooth(data, window_len=11)
                    plt.plot(data, linewidth=lw)
                    legend.append('#Units: ' + str(units[i]))
                plt.legend(legend, loc='lower right')
                plt.show()

            if 1:  # Loss vs epochs vs phase
                plt.figure()
                plt.grid()
                plt.title('Loss vs Epochs vs #Units')  # summarize history for accuracy
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                legend = ['Training Loss', 'Testing Loss']
                best_unit_id = 3
                # Training loss
                data = exp_data['history'][best_unit_id]['loss']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                # Validation loss
                data = exp_data['history'][best_unit_id]['val_loss']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                plt.legend(legend, loc='best')
                plt.show()

            if 1:  # Accuracy vs epochs vs phase
                plt.figure()
                plt.grid()
                plt.title('Accuracy vs Epochs vs #Units')  # summarize history for accuracy
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                legend = ['Training Accuracy', 'Testing Accuracy']
                best_unit_id = 3
                # Training acc
                data = exp_data['history'][best_unit_id]['acc']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                # Validation acc
                data = exp_data['history'][best_unit_id]['val_acc']
                data = smooth(data, window_len=11)
                plt.plot(data, linewidth=lw)
                plt.legend(legend, loc='best')
                plt.show()

    # Training loss vs optimizers
    if exp_02:
        exp_file_name = 'exp_02.npy'
        optimizers = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam']
        num_layers = 3
        num_units = 64

        if 0: # generate data
            all_history = []
            all_conf_matrix = []

            for i in range(len(optimizers)):
                print('#### Optimizer: {0:s}'.format(optimizers[i]))
                other_args = {'opt': optimizers[i]}
                history, conf_matrix = train_and_test(X, y, skf, n_epochs, n_folds, num_layers, num_units, other_args)
                all_history.append(history)
                all_conf_matrix.append(conf_matrix)
            exp_data = {'history': all_history, 'conf_matrix': all_conf_matrix}
            np.save(exp_file_name, exp_data)
        else: # Read data
            exp_data = np.load(exp_file_name).item()
            lw = 1

            if 1:  # Training Loss vs Epochs vs Optimizer
                plt.figure()
                plt.grid()
                plt.title('Training Loss vs Epochs vs Optimizer')  # summarize history for accuracy
                plt.ylabel('Training Loss')
                plt.xlabel('Epoch')
                legend = []
                for i in range(len(optimizers)):
                    data = exp_data['history'][i]['loss']
                    data = smooth(data, window_len=11)
                    plt.plot(data, linewidth=lw)
                    legend.append('Opt: ' + str(optimizers[i]))
                plt.legend(legend, loc='best')
                plt.show()


    # Get confusion matrix
    if exp_03:
        exp_file_name = 'exp_03.npy'
        num_layers = 3
        num_units = 64
        n_epochs = 200

        if 0:  # generate data
            other_args = {} # default optimizer is rmsprop
            history, conf_matrix = train_and_test(X, y, skf, n_epochs, n_folds, num_layers, num_units, other_args)
            all_history = []
            all_conf_matrix = []
            all_history.append(history)
            all_conf_matrix.append(conf_matrix)
            exp_data = {'history': all_history, 'conf_matrix': all_conf_matrix}
            np.save(exp_file_name, exp_data)
        else: # Read data
            exp_data = np.load(exp_file_name).item()
            print('Confusion Matrix: \n', exp_data['conf_matrix'][0])

