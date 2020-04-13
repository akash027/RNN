import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Embedding
from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import  pad_sequences
import os
import matplotlib.pyplot as plt


MAX_SEQ_LEN = 30
MAX_VOCAB_SIZE = 300
EPOCHS = 100
LATENT_DIM = 8
EMBEDDING_DIM = 300

encoder_que = []
decoder_que = []
decoder_ans = []

for lines in open("/home/sky/Documents/3.RNN & NLP/creating chat bot/chat.text"):
    que, ans = lines.split("\t")
    
    dec_q = '<sos> ' + ans
    dec_a = ans + ' <eos>'
    
    encoder_que.append(que)
    decoder_que.append(dec_q)
    decoder_ans.append(dec_a)


print(encoder_que)
print(decoder_que)
print(decoder_ans)



# Tokenization

tokenize = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenize.fit_on_texts(encoder_que+decoder_que+decoder_ans)

input_sequences = tokenize.texts_to_sequences(encoder_que)
target_input_sequences = tokenize.texts_to_sequences(decoder_que)
target_output_sequences = tokenize.texts_to_sequences(decoder_ans)
    


max_len_input = max(len(s) for s in input_sequences)

max_len_target = max(len(s) for s in target_input_sequences)


# get word to index mapping for output language

word2idx = tokenize.word_index
print("Found %s unique input tokens"%len(word2idx))

word2idx

num_words_output = len(word2idx)+1


# Padding

encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input, padding='post')
print("encoder_data shape: ", encoder_inputs.shape)
print("encoder_data[0] ", encoder_inputs[0])


decoder_inputs = pad_sequences(target_input_sequences, maxlen=max_len_target, padding='post')
print("encoder_data shape: ", decoder_inputs.shape)
print("encoder_data[0] ", decoder_inputs[0])

decoder_target = pad_sequences(target_output_sequences, maxlen=max_len_target, padding='post')


assert (word2idx['<sos>'])
assert (word2idx['<sos>'])

#store all the pre-trained word vectors
print('Loading word vectors...')

word2vec = {}

with open(os.path.join('/home/sky/Documents/3.RNN & NLP/creating chat bot/word_embeddings-1000 x 300.txt')) as f:
    # is just a space-seperated text file in the format:
    # word vec[0] vec[1] vec[2]... 

    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

print('Foundc %s word vectors.'%len(word2vec))




#preparing embedding matrix
print("Filling pre-trained embeddings...")

num_words = min(MAX_VOCAB_SIZE, num_words_output)
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
                            trainable = False)




# create targets

decoder_targets_one_hot = np.zeros((
    len(encoder_que),
    max_len_target,
    num_words_output),
    dtype='float32')


#assign the values
for i, d in enumerate(decoder_target):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1




## Model

encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embdeding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True)
encoder_outputs, h, c = encoder(x)

# keep only states, we need stated only for decoder
encoder_states = [h, c]


# set up the decoder, using [h,c] as initial state
decoder_inputs_placeholoder = Input(shape=(max_len_target,))

decoder_inputs_x = embdeding_layer(decoder_inputs_placeholoder)


#since decoder is many to many model we want to have return_sequences=True
# now in decoder we do not need hidden state and we will feed the hidden 
# state of encoder to decoder with decoder inputs

decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x,
                                     initial_state = encoder_states)


# final dense layer

decoder_dense = Dense(num_words_output, activation='softmax')  # total words are 72 (num_words_output) from that words our model will predict word word for each true word
decoder_outputs = decoder_dense(decoder_outputs)





# create model
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholoder], decoder_outputs)

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit([encoder_inputs,decoder_inputs], decoder_targets_one_hot,
          epochs=5000,
          validation_split=0.2)




#plot some data 

plt.plot(model.history.history['loss'], label='loss')
plt.plot(model.history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()



plt.plot(model.history.history['accuracy'], label='accuracy')
plt.plot(model.history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

#save model
#model.save("/home/sky/Documents/h5_model/chat.h5")



# from keras.models import load_model
# model = load_model("/home/sky/Documents/h5_model/chat.h5")



######## making predictions ######

# We need to create another model that can take in the RNN state and
# previous word as input and accept a T=1 sequence.


# The encoder will be a stand-alone
# From this we will get  our decoder hidden state


encoder_model = Model(encoder_inputs_placeholder, encoder_states)

deoder_state_input_h =  Input(shape=(LATENT_DIM,))
deoder_state_input_c =  Input(shape=(LATENT_DIM,))
decoder_states_inputs = [deoder_state_input_h, deoder_state_input_c]


decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = embdeding_layer(decoder_inputs_single)


# this time, we want to keep the states too, to be output
# by our sampling model

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x,
                                     initial_state=decoder_states_inputs)


decoder_states = [h, c]

decoder_outputs = decoder_dense(decoder_outputs)


# The sampling model
# inputs : y(t-1), h(t-1), c(t-1)
# outputs : y(t), h(t), c(t)

decoder_model = Model([decoder_inputs_single]+decoder_states_inputs,
                      [decoder_outputs]+decoder_states)




# map indexes back into real words
# so we can view the results

idx2word = {v:k for k, v in word2idx.items()}



def chat(question):
    
    question = tokenize.texts_to_sequences(question)
    question = pad_sequences(question,maxlen=max_len_input,padding='post')
    
    # Encode the input as state vector
    states_value = encoder_model.predict(question)
    
    
    #generate empty target sequence of length 1
    target_seq = np.zeros((1,1))
    
    
    #populate the first character of target sequence with the start character
    #Note: tokenizer lower-cases all words
    target_seq[0,0] = word2idx['<sos>']
    
    
    #if we get this we break
    eos = word2idx['<eos>']
    
    
    output_ = []
    
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict([target_seq]+states_value)
        
        
        #get the next word
        idx = np.argmax(output_tokens[0,0,:])
        
        #end sentence of EOS
        if eos == idx:
            break
        
        word = ''
        if idx > 0:
            word = idx2word[idx]
            output_.append(word)
        
        #update the decoder input
        target_seq[0,0] = idx
        
        #update state
        states_value = [h,c]
        
    return ' '.join(output_)



while True:
    que =  [input("ask question?")]
    answer = chat(que)
    print('-')
    print('ans: ', answer)
    ans = input("Continue? [y/n]")
    if ans and ans.lower().startswith('n'):
        break





