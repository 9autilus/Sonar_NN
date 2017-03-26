from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def create_model(input_shape, num_layers, num_units):
    m = Sequential()

    # First hidden layer
    m.add(Dense(num_units, activation='sigmoid', input_shape=input_shape))

    # Other hidden layers
    for i in range(1, num_layers):
        m.add(Dense(num_units, activation='sigmoid'))

    # Output layer
    m.add(Dense(1, activation='sigmoid'))

    return m


def train_model(m, model_name, X_train, y_train, X_test, y_test, n_epochs):
    # Optimizers
    opt = optimizers.RMSprop()
    # opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # Create check point callback
    checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_loss',
                                   verbose=0, save_best_only=True)

    h = m.fit(X_train, y_train, batch_size=32, nb_epoch=n_epochs,
              validation_data=(X_test, y_test), callbacks=[checkpointer],
               verbose=0)
    return h



def test_model(model_name, X):
    m = load_model(model_name)

    y = m.predict(X, batch_size=32)
    return y