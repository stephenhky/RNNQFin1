import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

# example from http://keras.io/examples/

def train_kerasLSTM(X_train, Y_train, timesteps=10):
    model = Sequential()
    model.add(Embedding(1, 256, input_length=1))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    model.fit(X_train, Y_train, batch_size=16, nb_epoch=timesteps)

    return model




#score = model.evaluate(X_test, Y_test, batch_size=16)