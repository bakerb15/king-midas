from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras import optimizers
import os

"""
This code was inspired by:
https://rubikscode.net/2018/03/26/two-ways-to-implement-lstm-network-using-python-with-tensorflow-and-keras/
"""


class Brain(object):

    def __init__(self, input_length, input_dimension, output_dim):

        model = Sequential()
        # input_shape = (input_length, input_dim)
        model.add(LSTM(input_shape=(input_length, input_dimension), return_sequences=True, units=(input_dimension)))
        # model.add(Dropout(0.2))
        # model.add(Dense(input_dimension))
        # model.add(Dense(input_dimension, activation='sigmoid', use_bias=True))
        # model.add(Dropout(0.2))
        # model.add(Dense(2* int(input_dimension), activation='sigmoid'))
        # model.add(LSTM(2*input_dimension))
        # model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(output_dim, activation='sigmoid'))
        opt = optimizers.Adagrad(lr=0.001, clipvalue=0.01, clipnorm=0.1)
        model.compile(loss='hinge',  optimizer=opt, metrics=['accuracy'])
        self.model = model

if __name__ == '__main__':
    path_to_data = './data'

    nn = Brain(12)
