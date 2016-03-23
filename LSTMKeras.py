import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence

# example from http://keras.io/examples/
# example 2: https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py

def train_kerasLSTM(X_train, Y_train, timesteps=10, max_features=1, max_len=1):
    # X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    model = Sequential()
    model.add(Embedding(max_features, 256, input_length=max_len))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    model.fit(X_train, Y_train, batch_size=32, nb_epoch=timesteps)

    return model




#score = model.evaluate(X_test, Y_test, batch_size=16)