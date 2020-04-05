import os
import sys
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score


#some Configuration

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5



'''
download the data: www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
download the word vectors : http://nlp.stanford.edu/data/glove.6b.zip
'''
#load in pre-trained word vectors
print('Loading word vectors...')

word2vec = {}

with open(os.path.join('/home/sky/Documents/3.RNN & NLP/toxic comment nlp rnn/glove.6B.%sd.txt' %EMBEDDING_DIM)) as f:
    # is just a space-seperated text file in the format:
    # word vec[0] vec[1] vec[2]....
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

print('Foundc %s word vectors.'%len(word2vec))


#prepare text samples and their labels
print('Loading in comments...')

train = pd.read_csv("/home/sky/Documents/3.RNN & NLP/toxic comment nlp rnn/toxic_cmnt/train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
targets = train[possible_labels].values



#convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)


# get word -> integer mapping
word2idx = tokenizer.word_index
print('Foundc %s unique tokens.'%len(word2idx))


# pad sequences
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('shape of data tensor:', data.shape)


#preparing embedding matrix
print("Filling pre-trained embeddings...")

num_words = min(MAX_VOCAB_SIZE, len(word2idx)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros
            embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an embdeding layer 
# note that we set trainable = False so as to keep the embeddings fixed

embdeding_layer = Embedding(num_words, EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable = False)


print("Building Model...")

#create an LSTM network with a single LSTM

input_ = Input(shape= (MAX_SEQUENCE_LENGTH,))
x = embdeding_layer(input_)
x = LSTM(15, return_sequences=True)(x)
# x = Bidirection(LSTM(15, return_sequences=True)(x)) # if u want to use bidirectional layer
x = GlobalMaxPooling1D()(x)

output_ = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(input_,output_)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])


print('Training Model...')

r = model.fit(data,targets,
              batch_size = BATCH_SIZE,
              epochs = EPOCHS,
              validation_split = VALIDATION_SPLIT,
              verbose = 1)


#plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


#accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()