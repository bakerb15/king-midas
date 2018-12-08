from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import os

"""
This code was inspired by:
https://rubikscode.net/2018/03/26/two-ways-to-implement-lstm-network-using-python-with-tensorflow-and-keras/
"""


class Brain(object):

    def __init__(self, input_length, input_dimension):

        model = Sequential()
        # input_shape = (input_length, input_dim)
        model.add(LSTM(input_shape=(input_length, input_dimension), return_sequences=True, units=(4*input_dimension)))
        model.add(Dropout(0.2))
        model.add(Dense(5*input_dimension))
        # model.add(Dense(int(input_dimension), activation='sigmoid'))
        model.add(LSTM(4*input_dimension))
        # model.add(Flatten())
        model.add(Dense(input_dimension, activation='softmax'))
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        self.model = model

if __name__ == '__main__':
    path_to_data = './data'

    nn = Brain(12)
