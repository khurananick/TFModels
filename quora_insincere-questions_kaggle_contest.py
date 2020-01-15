import sys
import numpy as np
from numpy import asarray
from numpy import zeros
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

np.random.seed(7)

df = pd.read_csv('../input/train.csv')
df.drop(['qid'],axis=1,inplace=True)

X = df.question_text
Y = df.target
le = LabelEncoder()
Y = le.fit_transform(Y).reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

max_words = 50000
embed_len = 300
max_len = 100
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

embeddings_index = dict()
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        line = None
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((max_words, embed_len))
for word, i in tok.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        try:
            embedding_matrix[i] = embedding_vector
        except:
            break

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,embed_len,weights=[embedding_matrix],input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(sequences_matrix,Y_train,batch_size=500,epochs=5,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

test_df = pd.read_csv('../input/test.csv')
X_test = test_df.question_text

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
predictions = model.predict(np.array(test_sequences_matrix), verbose=1)
predictions = [1 if prediction[0] > 0.25 else 0 for prediction in predictions]
submission = pd.DataFrame({'qid': test_df.qid, 'question_text': test_df.question_text, 'prediction':predictions}, columns=['qid','prediction'])
submission.to_csv('submission.csv', index=False)