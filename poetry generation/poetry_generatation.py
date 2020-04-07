import os
import sys
import string
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.optimizers import Adam, SGD


'''
download the data: any text data like poetry , essays etc
download the word vectors : http://nlp.stanford.edu/data/glove.6b.zip
'''


#Some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 100
LATENT_DIM = 25

# LOAD IN DATA 

input_texts = []
target_texts = []

for line in open("/home/sky/Documents/3.RNN & NLP/poetry generation/poetry.txt"):
    line = line.rstrip()
    if not line:
        continue
    
    input_line = '<sos> ' + line
    target_line = line + ' <eos>'
    
    input_texts.append(input_line)
    target_texts.append(target_line)

all_lines = input_texts + target_texts


# Convert the sentences (string) into integers

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)

input_sequnces = tokenizer.texts_to_sequences(input_texts)
target_sequnces = tokenizer.texts_to_sequences(target_texts)

#find max seq len 
max_sequence_length_from_data = max(len(s) for s in input_sequnces)
print("max sequence length: ", max_sequence_length_from_data)


#get word -> integer mapping

word2idx = tokenizer.word_index
print('Found %s unique tokens.'%len(word2idx))

assert('<sos>' in word2idx)
assert('<eos>' in word2idx)


# Pad sequences so that we  get N x T matrix

max_sequence_length = min(max_sequence_length_from_data,MAX_SEQUENCE_LENGTH)
input_sequnces = pad_sequences(input_sequnces, maxlen=max_sequence_length,padding='post')
target_sequnces = pad_sequences(target_sequnces, maxlen=max_sequence_length,padding='post')



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




# one hot the targets (cant use sparse cross-entropy)

one_hot_targets = np.zeros((len(input_sequnces), max_sequence_length, num_words))

for i, target_sequnce in enumerate(target_sequnces):
    for t, word in enumerate(target_sequnce):
        if word > 0:
            one_hot_targets[i,t,word] = 1



# load pre-trained word embeddings into an embdeding layer 
# note that we set trainable = False so as to keep the embeddings fixed

embdeding_layer = Embedding(num_words, EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            trainable = False)


print("Building Model....")


#create an LSTM network with a single LSTM

input_ = Input(shape = (max_sequence_length,))
initial_h = Input(shape = (LATENT_DIM,))
initial_c = Input(shape = (LATENT_DIM,))

x = embdeding_layer(input_)

lstm = LSTM(LATENT_DIM, return_state=True, return_sequences=True)
x,_,_ = lstm(x, initial_state=[initial_h,initial_c],) # dont need the states here

dense = Dense(num_words, activation='softmax')
output_ = dense(x)

model = Model([input_, initial_h, initial_c], output_)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])

print(model.summary())



print("training model...")

z = np.zeros((len(input_sequnces), LATENT_DIM))
r = model.fit([input_sequnces, z, z], one_hot_targets,
              batch_size=BATCH_SIZE,epochs=EPOCHS,
              validation_split=VALIDATION_SPLIT)



# making a sampling model

input2 = Input(shape=(1,)) # we will only input one word at a time 
x = embdeding_layer(input2)

x, h, c = lstm(x, initial_state=[initial_h,initial_c]) # now we need states to feed in

output2 = dense(x)

sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])



#reverse word2idx dictionary to get back words
# during prediction
idx2word = {v:k for k, v in word2idx.items()}



def sample_line():
    # initial inputs
    
    np_input = np.array([[ word2idx['<sos>'] ]])
    h = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))
    
    #so we know when to quit
    eos = word2idx['<eos>']
    
    # store the output sentence here
    output_sentence = []
    
    for _ in range(max_sequence_length):
        o, h, c = sampling_model.predict([np_input,h, c])
        
        #print("o.shape: ", o.shape, o[0,0,:10])
        #idx = np.argmax(o[0,0])
        
        probs = o[0,0]
        if np.argmax(probs) == 0:
            print("wtf")
        probs[0] = 0
        
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        
        if idx == eos:
            break
        
        #acculate output
        output_sentence.append(idx2word.get(idx, '<WTF %d>' % idx))
        
        
        #make the next input into model
        np_input[0,0] = idx
        
    
    return ' '.join(output_sentence)
        




# generate a 4 line poem
    
while True:
    for _ in range(4):
        print(sample_line())
        
    ans = input("-----generate another? [y/n]------")
    if ans and ans[0].lower().startswith('n'):
        break
        




