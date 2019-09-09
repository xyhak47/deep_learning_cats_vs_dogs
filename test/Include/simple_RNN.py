
"""
state_t = 0
for input_t in input_sequence:
    output_t = f(input_t, state_t)
    state_t = output_t


state_t = 0
for input_t in input_sequence:
    # output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
"""


import numpy as np

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# c = a.copy()
# d = a * c
# e = np.dot(a, c)
# f = np.cross(a, c)



'''simple RNN by numpy'''
timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))  # 100x32

state_t = np.zeros((output_features))  # 64

W = np.random.random((output_features, input_features))   # 64x32
U = np.random.random((output_features, output_features))  # 64x64
b = np.random.random((output_features))  # 64

successive_outputs = []
for input_t in inputs:  # input_t shape is (input_features,)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)





'''
simpleRNN in keras
'''
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))  # 返回最终结果
model.summary()


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))  # 返回每个时序的结果
model.summary()



model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True)) # 中间多个堆叠
model.add(SimpleRNN(32, return_sequences=True)) # 中间多个堆叠
model.add(SimpleRNN(32, return_sequences=True)) # 中间多个堆叠
model.add(SimpleRNN(32))        # 最后一层仅输出结果
model.summary()




'''imdb demo
'''
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (sample x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
histroy = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

from draw_data import *

draw_data(histroy)



'''
LSTM
'''
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
histroy = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

draw_data(histroy)
