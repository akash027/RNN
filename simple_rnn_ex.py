from __future__ import print_function
from builtins import range,input

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, GRU
import matplotlib.pyplot as plt

#length (sequence)
T = 8

#input dimensionality 
D = 2

#Hidden layer size
M = 3

X = np.random.randn(1,T,D)


def lstm1():
    input_ = Input(shape=(T,D))
    rnn = LSTM(M, return_state = True)
    x = rnn(input_)
    
    model = Model(input_,x)
    o,h,c = model.predict(X)
    print("o:", o)
    print("h:", h)
    print("c:", c)

def lstm2():
    input_ = Input(shape=(T,D))
    rnn = LSTM(M, return_state = True, return_sequences=True)
    x = rnn(input_)
    
    model = Model(input_,x)
    o,h,c = model.predict(X)
    print("o:", o)
    print("h:", h)
    print("c:", c)


def gru1():
    input_ = Input(shape=(T,D))
    rnn = GRU(M, return_state = True)
    x = rnn(input_)
    
    model = Model(input_,x)
    o,h = model.predict(X)
    print("o:", o)
    print("h:", h)
    

def gru2():
    input_ = Input(shape=(T,D))
    rnn = GRU(M, return_state = True, return_sequences=True)
    x = rnn(input_)
    
    model = Model(input_,x)
    o,h = model.predict(X)
    print("o:", o)
    print("h:", h)


print(lstm1())
print(lstm2())
print(gru1())
print(gru2())
