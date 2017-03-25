import numpy as np


if __name__ == '__main__':
    x_raw = np.genfromtxt(r'.\..\data\sonar.all-data.csv', delimiter=',', dtype='str')
    X = x_raw[:, :x_raw.shape[1]-1].astype('float')
    y = (x_raw[:, x_raw.shape[1]-1] == 'R').astype(int)



    exit(0)