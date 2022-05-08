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

############################################
########CNN Mental Health Profile###########


name = 'WiredHealth'
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

def update(name):
    sheet_id = '1LbU-4BxEyfBFqI5A4RjiFtzDzxHCbJeDrcHSRViF--w'
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=1638373022"
    df =pd.read_csv(url)
    df.to_csv(f"PFD2/{name}.csv",index_label="Index")

def data_cleaning(name):
    columnsMap={
        "Roughly how much time do you spend exercising this week": "activeness","How much sleep did you get today ":"sleepTime",
        "Rate your sleep from 1-10 ":"sleepQuality","How much time roughly did you spend on your screen today? (Only Mobile Devices)":"mobileDeviceUsage",
        "How much time did you spend on social media applications today? (Facebook, Instagram, etc)":"socialMediaUsage",
        "How overwhelmed are you?":"overwhelmedLevel","How emotionally exhausted are you?":"emotionalExhaustionLevel","How stressed are you? 😁 - 🤬":"stressLevel",
        "One sentence to describe how you feel (e.g. I feel ...  because ...... )":"statement"
        }
    rowMapList = {
        "Gender":{"Male":0,"Female":1, "Other":2},
        "Age Group":{"<21":0, "21< age < 31":1 ,"31< age < 41":2},
        "activeness":{"None 😴":0, "Under 30 minutes":1, "30 - 60 minutes":2, "1 hour - 2 hours":3, "2 hour and above 💪":4},
        "sleepTime":{"0 - 1 hour 😑":0, "2-4 hours":1, "5-7 hours":2, "8-10 hours":3, "10 hours and above 😴":4},
        "mobileDeviceUsage":{"None 😧":0, "Less than an hour":1, "1-2 hours":2, "4-5 hours":3, "7-8 hours":4, "More than 10 hours":5},
        "socialMediaUsage":{"Less than an hour":0, "1-2 hours":1, "3 -4 hours":2, "More than 4 hours":3}
    }
    df = pd.read_csv(f"PFD2/{name}.csv")
    df.rename(columns=columnsMap,inplace=True)
    for row in rowMapList.keys():
        df[row].replace(rowMapList[row], inplace=True)
    sentimentList = [SentimentIntensityAnalyzer().polarity_scores(x)['compound'] for x in df['statement']]
    df['vaderSentiments'] = sentimentList
    testimonialList = [TextBlob(x).sentiment[0] for x in df['statement']]
    df['blobSentiments'] = testimonialList
    featuresdf = df[["Gender","Age Group","activeness","sleepTime","mobileDeviceUsage","socialMediaUsage","vaderSentiments","blobSentiments"]]
    predStressDf = df["stressLevel"]
    predOWDf = df["overwhelmedLevel"]
    predEEDf = df["emotionalExhaustionLevel"]
    x_train, x_test, y_train, y_test = train_test_split(featuresdf,predOWDf, test_size=0.20, random_state=1)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    return(x_train, x_test, y_train_one_hot, y_test_one_hot)

def run_model():
    x_train, x_test, y_train, y_test = data_cleaning(name)
    model = Sequential()
    model.add(Dense(8,activation='relu',input_shape = (8,)))
    model.add(Dense(36,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(81,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','crossentropy'])
    K.set_value(model.optimizer.learning_rate, 0.0001)
    model.summary()
    hist = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs = 1000, callbacks =[earlystopping])
    return model


#update(name)
model = run_model()

#pred = model.predict(values)
#print(pred)