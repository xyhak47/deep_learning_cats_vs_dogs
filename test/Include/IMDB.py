import os
from path_generator import root_path
from path_generator import join_path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from draw_data import *


#34kb,84gb,210dt,216kb/s,total:18456

talijahwgrzdu
whoisyourdaddy

imdb_dir = join_path(root_path, 'aclImdb')
# print(imdb_dir)
train_dir = join_path(imdb_dir, 'train')
# print(train_dir)



def append_data(dir):
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = join_path(dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(join_path(dir_name, fname), encoding='UTF-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return labels, texts



labels, texts = append_data(train_dir)


# print(texts)
# print(labels)

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)

# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]



glove_dir = join_path(root_path, 'glove.6B')

embeddings_index = {}
f = open(join_path(glove_dir, 'glove.6B.100d.txt'), encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    # print(len(values))
f.close()

# print('Found %s word vectors.' % len(embeddings_index))

embeddings_dim = 100

embeddings_matrix = np.zeros((max_words, embeddings_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(max_words, embeddings_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

model.layers[0].set_weights([embeddings_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
histroy = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
pre_train_glove_model_path = join_path(imdb_dir, 'pre_train_glove_model.h5')
model.save_weights(pre_train_glove_model_path)

# draw_data(histroy)


test_dir = join_path(imdb_dir, 'test')

labels, texts = append_data(test_dir)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model.load_weights(pre_train_glove_model_path)
model.evaluate(x_test, y_test)
