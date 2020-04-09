import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

'''
download the word vectors : http://nlp.stanford.edu/data/glove.6b.zip
'''


#some configuration
BATCH_SIZE = 64  #batch size for training
EPOCHS = 10
LATENT_DIM = 256  #latent dimension of the encoding space
NUM_SAMPLES = 10000  #num of samples to train on 
MAX_SEQUENCE_LENGTH = 100 
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 50


#where we will store the data

input_texts = []  #sentence in original language
target_texts = []  #sentence in target language
target_texts_inputs = []   #sentence in target language offset by 1



#load in data 

t = 0
for line in open("/fra-eng/fra.txt"):
    #only keep a limited number of samples
    
    t += 1
    if t > NUM_SAMPLES:
        break

    
    # input  and target are seperated bt atb
    if '\t' not in line:
        continue
    
    #split up the input and translation
    input_text, translation, _ = line.split('\t')
    
    
    # make the target input and output
    # recall we will be using teacher forcing
    target_text_input = '<sos> ' + translation
    target_text = translation + ' <eos>'
    
    input_texts.append(input_text)
    target_texts_inputs.append(target_text_input)
    target_texts.append(target_text)
    

print("num samples: ", len(input_texts))



# tokenize the inputs

tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)


#get the word to index mappping for input language

word2idx_inputs = tokenizer_inputs.word_index
print("Found %s unique input tokens"%len(word2idx_inputs))

# determine maximum length input sequence

max_len_input = max(len(s) for s in input_sequences)



# tokenize the outputs
# dont filter out special characters
# otherwise <sos> and <eos> wont appear

tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)

target_sequences_inputs =  tokenizer_outputs.texts_to_sequences(target_texts_inputs)
target_sequences =  tokenizer_outputs.texts_to_sequences(target_texts)


# get word to index mapping for output language

word2idx_outputs = tokenizer_outputs.word_index
print("Found %s unique input tokens"%len(word2idx_outputs))


# store number of output words for later
# remember to add 1 sinx=ce indexing start at 1

num_words_output = len(word2idx_outputs)+1


# determine maximum length output sequence

max_len_target = max(len(s) for s in target_sequences)
 

# pad the sequence

encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input, padding='post')
print("encoder_data shape: ", encoder_inputs.shape)
print("encoder_data[0] ", encoder_inputs[0])



decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("encoder_data shape: ", decoder_inputs.shape)
print("encoder_data[0] ", decoder_inputs[0])


decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')



#store all the pre-trained word vectors
print('Loading word vectors...')

word2vec = {}

with open(os.path.join('/glove.6B.%sd.txt' %EMBEDDING_DIM)) as f:
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

num_words = min(MAX_NUM_WORDS, len(word2idx_inputs)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx_inputs.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros
            embedding_matrix[i] = embedding_vector


# load pre-trained word embeddings into an embdeding layer 
# note that we set trainable = False so as to keep the embeddings fixed

embdeding_layer = Embedding(num_words, EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            trainable = False)




# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences

decoder_targets_one_hot = np.zeros((len(input_texts),
                                    max_len_target,
                                    num_words_output),
                                   dtype='float32')

print(decoder_targets_one_hot.shape)


# assign the values 

for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1
        

        


########### Bulding the model ######

encoder_inputs_placeholder = Input(shape = (max_len_input,))

x = embdeding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)
encoder_outputs, h, c = encoder(x)
## encoder_outputs, h, = encoder(x)  ##if GRU used


#keep only the states to pass into decoder

encoder_states = [h, c]
## encoder_states = [states_h] #gru


#set up the decoder, using [h,c] as initial state

decoder_inputs_placeholder = Input(shape=(max_len_target,))

#this word embedding will not use pretrained vectors
#although you could

decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)


#since the decoder is a 'to-many' model we want to have
#return _sequences =True

decoder_lstm = LSTM(LATENT_DIM, return_state=True, return_sequences=True, dropout=0.5)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)


## decoder_outputs = decoder_gru(decoder_inputs_x, initial_state=encoder_states)


#final dense layer for prediction

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


#create  the model object

model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder],decoder_outputs)


print(model.summary())

#compile and train model

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


r = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot,
              batch_size =BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2)




# plot some data 

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()



plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()


#save model
model.save("/translation.h5")



######## making predictions ######

# We need to create another model that can take in the RNN state and
# previous word as input and accept a T=1 sequence.


# The encoder will be a stand-alone
# From this we will get  our decoder hidden state

encoder_model = Model(encoder_inputs_placeholder, encoder_states)


decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]
#decoder_states_inputs = [decoder_state_input_h] #gru


decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# this time, we want to keep the states too, to be output
# by our sampling model

decoder_outputs, h, c =  decoder_lstm(decoder_inputs_single_x,
                                      initial_state=decoder_states_inputs)

# decoder_outputs, state_h =  decoder_lstm(decoder_inputs_single_x,
#                                       initial_state=decoder_states_inputs)
#gru


decoder_states = [h, c]
#decoder_states = [h]

decoder_outputs = decoder_dense(decoder_outputs)


# The sampling model
# inputs : y(t-1), h(t-1), c(t-1)
# outputs : y(t), h(t), c(t)

decoder_model = Model([decoder_inputs_single]+decoder_states_inputs,
                      [decoder_outputs]+decoder_states)


# map indexes back into real words
# so we can view the results

idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}



def decode_sequence(input_seq):
    
    #Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)
    
    
    #generate empty target sequence of length 1
    target_seq = np.zeros((1,1))
    
    #populate the first character of target sequence with the start character
    #Note: tokenizer lower-cases all words
    target_seq[0,0] = word2idx_outputs['<sos>']
    
    #if we get this we break
    eos = word2idx_outputs['<eos>']
    
    #Create the translation
    output_sentence = []
    
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict([target_seq]+
                                                    states_value)
        #output_tokens, h = decoder_model.predict([target_seq]+
        #                                               states_value)
        #gru
        
        
        
        #Get next word
        idx = np.argmax(output_tokens[0,0,:])
        
        #End sentence of EOS
        if eos == idx:
            break
        
        word = ''
        
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)
            
            
        
        # Update the decoder input
        # Which is just the word just generated
        target_seq[0,0] = idx
        
        #Update states
        states_value = [h,c]
        #states_value = [h] #gru
        
    return ' '.join(output_sentence)



while True:
    
    #do some test translations
     i = np.random.choice(len(input_texts))
     input_seq = encoder_inputs[i:i+1]
     translation = decode_sequence(input_seq)
     print('-')
     print('Input: ', input_texts[i])
     print('translation: ', translation)
     
     ans = input("Continue? [y/n]")
     if ans and ans.lower().startswith('n'):
         break







