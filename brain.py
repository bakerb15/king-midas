from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import os

"""
This code was inspired by:
https://rubikscode.net/2018/03/26/two-ways-to-implement-lstm-network-using-python-with-tensorflow-and-keras/
"""


class Brain(object):

    def __init__(self, input_dimension):

        model = Sequential()
        model.add(Dense(2*input_dimension, input_dim=input_dimension))
        model.add(Dropout(0.2))
        model.add(LSTM(3*input_dimension, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(int(input_dimension), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

if __name__ == '__main__':
    path_to_data = './data'

    nn = Brain(12)
