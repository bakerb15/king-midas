from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, TimeDistributed
from keras import optimizers
import numpy as np
import keras.backend as K

import os

"""
This code was inspired by:
https://rubikscode.net/2018/03/26/two-ways-to-implement-lstm-network-using-python-with-tensorflow-and-keras/
"""


class Brain(object):

    # def __init__(self, input_length, input_dimension, output_dim):
    #
    #     model = Sequential()
    #     # input_shape = (input_length, input_dim)
    #     model.add(LSTM(input_shape=(input_length, input_dimension), return_sequences=True, units=(input_dimension)))
    #     # model.add(Dropout(0.2))
    #     model.add(Dense(5*input_dimension, use_bias=True, activation='sigmoid') )
    #     # model.add(Dense(input_dimension, activation='sigmoid', use_bias=True))
    #     model.add(Dropout(0.2))
    #     # model.add(Dense(2* int(input_dimension), activation='sigmoid'))
    #     model.add(LSTM(4*input_dimension, return_sequences=True, use_bias=True))
    #     model.add(Dropout(0.2))
    #     model.add(Flatten())
    #     model.add(Dense(output_dim, activation='tanh'))
    #     sgd = optimizers.Adam(lr=0.01 )
    #     model.compile(loss='mean_squared_error',  optimizer=sgd, metrics=['accuracy'])
    #     self.model = model

    # was trying to predict one company price
    # def __init__(self, input_length, input_dimension, output_dim):
    #
    #     model = Sequential()
    #     # input_shape = (input_length, input_dim)
    #     model.add(LSTM(input_shape=(input_length, input_dimension), return_sequences=True, units=(input_dimension)))
    #     # model.add(Dropout(0.2))
    #     model.add(Dense(5*input_dimension, use_bias=True, activation='sigmoid') )
    #     # model.add(Dense(input_dimension, activation='sigmoid', use_bias=True))
    #     model.add(LSTM(int(input_dimension), return_sequences=False))
    #     #model.add(LSTM(4*input_dimension, return_sequences=True, use_bias=True))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(output_dim, activation='linear'))
    #     sgd = optimizers.Adam(lr=0.1)
    #     model.compile(loss='mae',  optimizer='rmsprop', metrics=['accuracy'])
    #     self.model = model

    #a
    # def __init__(self, input_length, input_dimension, output_dim):
    #
    #     model = Sequential()
    #     # input_shape = (input_length, input_dim)
    #     model.add(LSTM(input_shape=(input_length, input_dimension), return_sequences=True, units=(input_dimension)))
    #     model.add(Dropout(0.2))
    #     # model.add(Dense(5*input_dimension, use_bias=True, activation='sigmoid') )
    #     model.add(Dense(4   *input_dimension, activation='tanh', use_bias=True))
    #     #model.add(LSTM(int(input_dimension), return_sequences=False))
    #     #model.add(LSTM(4*input_dimension, return_sequences=True, use_bias=True))
    #     model.add(Dropout(0.2))
    #     model.add(Flatten())
    #     model.add(Dense(output_dim, activation='relu'))
    #     sgd = optimizers.Adam(lr=0.1)
    #     model.compile(loss='categorical_crossentropy',  optimizer='adadelta', metrics=['accuracy'])
    #     self.model = model

    #b2
    # def __init__(self, input_length, input_dimension, output_dim):
    #     model = Sequential()
    #     # input_shape = (input_length, input_dim)
    #     model.add(Dense(input_shape=(input_length, input_dimension), activation='sigmoid', units=(input_dimension)))
    #     model.add(Dropout(0.2))
    #     # model.add(Dense(5*input_dimension, use_bias=True, activation='sigmoid') )
    #     model.add(Flatten())
    #     model.add(Dense(output_dim, activation='relu', use_bias=True))
    #     # model.add(LSTM(int(input_dimension), return_sequences=False))
    #     #model.add(LSTM(4*input_dimension, return_sequences=False, use_bias=True))
    #     # model.add(Dense(output_dim, activation='relu'))
    #     #model.add(Dense(output_dim, activation='sigmoid'))
    #     sgd = optimizers.Adam(lr=0.1)
    #     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #     self.model = model

    #c
    # def __init__(self, input_length, input_dimension, output_dim):
    #     model = Sequential()
    #     # input_shape = (input_length, input_dim)
    #     model.add(LSTM(input_shape=(input_length, input_dimension), return_sequences=True, units=(input_dimension)))
    #     model.add(TimeDistributed(Dense(input_dimension)))
    #     # model.add(Dense(5*input_dimension, use_bias=True, activation='sigmoid') )
    #     model.add(Dense(4 * input_dimension, activation='tanh', use_bias=True))
    #     # model.add(LSTM(int(input_dimension), return_sequences=False))
    #     #model.add(LSTM(4*input_dimension, return_sequences=False, use_bias=True))
    #     model.add(Dropout(0.4))
    #     #model.add(Flatten())
    #     # model.add(Dense(output_dim, activation='relu'))
    #     #model.add(Dense(output_dim, activation='sigmoid'))
    #     model.add(LSTM(output_dim, return_sequences=False, activation='sigmoid'))
    #     sgd = optimizers.Adam(lr=0.1)
    #     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #     self.model = model

    # d2
    def __init__(self, input_length, input_dimension, output_dim):
        model = Sequential()
        model.add(LSTM(input_shape=(input_length, input_dimension), recurrent_activation='relu',activation='relu', return_sequences=True, units=(input_dimension)))
        model.add(LSTM(int(input_dimension/2), use_bias=True, recurrent_activation='relu',activation='relu', return_sequences=True))
        model.add(LSTM(int(input_dimension/4), use_bias=True, recurrent_activation='relu',activation='relu', return_sequences=False))
        model.add(Dense(output_dim, activation='relu'))
        opt = optimizers.RMSprop(lr=0.01, clipvalue=1.0)
        model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])
        self.model = model


class Brain2(object):

    # #sp2
    # def __init__(self, input_length, input_dimension, output_dim):
    #     model = Sequential()
    #     model.add(LSTM(input_shape=(input_length, input_dimension), recurrent_activation='tanh',activation='relu', return_sequences=False, units=(2*input_dimension)))
    #     model.add(Dense(output_dim, activation='tanh'))
    #     opt = optimizers.RMSprop(lr=0.01, clipvalue=1.0)
    #     model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
    #     self.model = model

    #sp4
    def __init__(self, input_length, input_dimension, output_dim):
        model = Sequential()
        model.add(LSTM(input_shape=(input_length, input_dimension), recurrent_activation='tanh', activation='tanh', use_bias=True, return_sequences=False, units=(input_dimension)))
        model.add(Dense(output_dim, activation='tanh', use_bias=True))
        opt = optimizers.RMSprop(lr=0.01, clipvalue=1.0)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        self.model = model

    # #sp5
    # def __init__(self, input_length, input_dimension, output_dim):
    #     model = Sequential()
    #     model.add(LSTM(input_shape=(input_length, input_dimension), recurrent_activation='tanh',activation='tanh', use_bias=True, return_sequences=False, units=(2*input_dimension)))
    #     # model.add(LSTM(2*int(input_dimension), use_bias=True, recurrent_activation='tanh', activation='tanh', return_sequences=True))
    #     # model.add(LSTM(int(input_dimension), use_bias=True, recurrent_activation='tanh', activation='tanh', return_sequences=False))
    #     model.add(Dense(output_dim, activation='tanh', use_bias=True))
    #     opt = optimizers.RMSprop(lr=0.1)
    #     model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    #     self.model = model



if __name__ == '__main__':
    pass
