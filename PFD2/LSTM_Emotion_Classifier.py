import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.layers import Dense,Dropout
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras import callbacks
import pandas as pd
from string import punctuation
import nltk 
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import models,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional

############################################
#########LSTM Emotion Classifier############

train_df = pd.read_csv("PFD2/train.txt",sep = ";", names=['text', 'emotions'])
test_df = pd.read_csv("PFD2/test.txt",sep = ";", names=['text', 'emotions'])
stopword = nltk.corpus.stopwords.words('english')

wn = nltk.WordNetLemmatizer()
def clean_text(word):
    withoutPunct = ''.join([letter for letter in word if letter not in punctuation])
    tokens = re.split('\W+', withoutPunct)
    return([wn.lemmatize(word.lower()) for word in tokens if tokens not in stopword])

#Encode X & Y
le = LabelEncoder()
x_train = train_df['text'].apply(lambda x: clean_text(x))
x_test = test_df['text'].apply(lambda x: clean_text(x))
y_train = to_categorical(le.fit_transform(train_df["emotions"]))
y_test = to_categorical(le.transform(test_df["emotions"]))
print(y_train)
print(train_df["emotions"])

df = pd.concat([x_train, x_test], axis=0)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df)

sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

print(x_train)

#Only use the first 256 words
x_train = pad_sequences(sequences_train, maxlen=256, truncating='pre')
x_test = pad_sequences(sequences_test, maxlen=256, truncating='pre')
print(x_train)

vocabSize = len(tokenizer.index_word) + 1


inputs = Input(shape=(None,), dtype="int32")
model = Sequential()
model.add(Embedding(20000, 128,  input_length=x_train.shape[1]))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy','crossentropy'])
K.set_value(model.optimizer.learning_rate, 0.001)
model.summary()
hist = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs = 3)

# Saving model structure to a JSON file 
model_json = model.to_json() 
with open("PFD2/network.json", "w") as json_file:
    json_file.write(model_json)
# Saving weights of the model to a HDF5 file
model.save_weights("PFD2/network.h5")
with open('PFD2/tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
    
with open('PFD2/labelEncoder.pickle', 'wb') as f:
    pickle.dump(le, f)

