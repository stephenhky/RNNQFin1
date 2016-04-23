'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.models import model_from_json

import argparse

import pandas as pd

def argparser():
    parser = argparse.ArgumentParser(description='LSTM Keras example')
    parser.add_argument('modelprefix', help='path of model prefix')
    parser.add_argument('--toload', action='store_true', help='load model instead of train')
    parser.add_argument('--testoutput', default='', help='test output data')
    return parser

def save_model(nameprefix, model):
    model_json = model.to_json()
    open(nameprefix+'.json', 'wb').write(model_json)
    model.save_weights(nameprefix+'.h5')

def load_model(nameprefix):
    model = model_from_json(open(nameprefix+'.json', 'rb').read())
    model.load_weights(nameprefix+'.h5')
    return model

# parse argument
parser = argparser()
args = parser.parse_args()


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

if not args.toload:
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    print(X_train.shape)
    # print(y_train.shape)
    print(len(y_train))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
              validation_data=(X_test, y_test))

    save_model(args.modelprefix, model)
else:
    print("Loading model")
    model = load_model(args.modelprefix)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

if len(args.testoutput)>0:
    Y_predicted = model.predict(X_test, batch_size=batch_size)
    # df = pd.DataFrame({'y_test': y_test,
    #                    'Y_predicted': Y_predicted})
    # df.to_csv(args.testoutput, index=False)
    print(Y_predicted)
    print(y_test)