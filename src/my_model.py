from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

def create_model(input_shape):
    m = Sequential()
    m.add(Dense(8, activation='sigmoid', input_shape=input_shape))
    m.add(Dense(1, activation='sigmoid'))

    return m




def train_model(m, X, y, n_epochs):
    # Optimizer
    rms = RMSprop(lr=0.001, decay=0.1)
    m.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(X, y, batch_size=1, nb_epoch=n_epochs)
    return m



def test_model(m, X):
    y = m.predict(X, batch_size=32)
    return y