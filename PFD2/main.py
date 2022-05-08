from logging import debug
from telegram.ext import Updater,CommandHandler,MessageHandler,Filters,ConversationHandler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import random
from datetime import date
import os
import nltk 
import re
from time import sleep
from json import load
from string import punctuation
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

############################################
##############Telegram APP##################

nltk.download('wordnet')
nltk.download('stopwords')

print("Bot started...")

# Define variables
bot = "BOT: "
user = "USER: "
stopword = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()
name = "Mental Health bot"
weather = "cloudy"

userID = None
database = "PFD2/Data.csv"
signup = False
debugMode = False

API_KEY = '2121556065:AAFq0vvkKpdHJYg9CGHQ7xGYCYG49Taz-hQ'
NAME, AGE, PROFESSION, TIME, ACTIVENESS = range(5)
ID,PASSWORD,SUCCESS,FAIL = range(4)
LOGINID = 0

# Loading pickle files
with open('PFD2/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('PFD2/labelEncoder.pickle', 'rb') as f:
    le = pickle.load(f)
# Loading JSON file 
json_file = open("PFD2/network.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Loading weights
loaded_model.load_weights("PFD2/network.h5")

# Return the matching response if there is one, default otherwise
def backendData(message):
    emotion = sentiment_scores(message)
    df = pd.read_csv("PFD2/Data.csv",index_col="Index")
    totale = df[df["ID"] == userID]["Compound Score"].to_list()[0]
    r,p = prediction(message)
    print(r,p)
    return(f"Text Polarization: {emotion}\nToday's Polarization: {totale:.3f}\nText Emotion: {r}\nProbability: {p*100:.1f}%")

def prediction(text):
    x_test = clean_text(text)
    sequences_train = tokenizer.texts_to_sequences([x_test])
    x_test = pad_sequences(sequences_train, maxlen=256, truncating='pre')
    result = le.inverse_transform(np.argmax(loaded_model.predict(x_test), axis=-1))[0]
    predict_x =  np.max(loaded_model.predict(x_test))
    return(result, predict_x)

def clean_text(word):
    withoutPunct = ''.join([letter for letter in word if letter not in punctuation])
    tokens = re.split('\W+', withoutPunct)
    return([wn.lemmatize(word.lower()) for word in tokens if tokens not in stopword])

def respond_sentiment(message,name):
    # Check if the message is in the response
    emotion = sentiment_scores(message)
    if emotion == (None or 0 or 0.0):
      respond = {
            0:"How are you feeling now {0}".format(name),
            1:"Hmm tell me more!",
            2:"Well {0}, what else is going on?".format(name),
            3:"Hey {0} What can I do for you today ?".format(name)
                }
      num = random.randint(0,3)
      bot_message = respond[num]
      return bot_message
    elif emotion > 0:
      respond_pos = {
          "0.1":"Congrats & hey try to enjoy yourself more !?",
          "0.2":"Thats interesting to hear",
          "0.3": "Wow thats interesting",
          "0.4": "How nice to hear that from you {0}".format(name),
          "0.5": "Well {0} im happy for you".format(name),
          "0.6": "Hey thats great {0}".format(name),
          "0.7":" Yo thats great {0} tell me more !".format(name),  
          "0.8": "OMG! That sounds so fun, im interested to know more ",
          "0.9": "I only wish i was with you :) It Sounds so fun",
          "1.0": "Man it would be boring if i didnt get to hear more ;-;"
                }
      list_of_pos = ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
      for x in range(len(list_of_pos)):
        if list_of_pos[x] in str(emotion):
          res_pos = list_of_pos[x]
          bot_message = respond_pos[res_pos]
          return bot_message
    elif emotion < 0:
      respond_neg = {
          "-0.1":"Do you want to talk about it?",
          "-0.2":"Thats interesting... maybe you want to talk more about it?",
          "-0.3":"hmm why do you think like that?",
          "-0.4": "Hey dont think like that {0}".format(name),
          "-0.5": "Please think before doing anything you will regret".format(name),
          "-0.6": "Well {0}, the world is full is dissapointments but know you are not one of it".format(name),
          "-0.7": "Hey now {0} tell me more, im here to listen to you please ask for help if you need it".format(name),  
          "-0.8": "Gosh that sounds harsh please don't do anything rash ",
          "-0.9": "Please be careful, know that there are people here for you, no matter what you do not suffer alone",
          "-1.0": "{0}, Please don't do anything you will regret, there are people in this world who love you and i am sure that even if they are not here now there will be people who will come to love you"
                    }
      list_of_neg = ["-0.1","-0.2","-0,3","-0.4","-0.5","-0.6","-0.7","-0.8","-0.9","-1.0"]
      for x in range(len(list_of_neg)):
        if list_of_neg[x] in str(emotion) :
          res_neg = list_of_neg[x]
          bot_message = respond_neg[res_neg]
          return bot_message

def adding_to_compound(df,score,text):
    today = str(date.today())
    buffer = 0.1
    #df = df.astype('object')
    compoundscore = score
    if score >= buffer:
        neg = ""
        pos = text
    elif score <= -buffer:
        neg = text
        pos = ""
    else:
        neg = ""
        pos = ""
    row = pd.DataFrame([[today,compoundscore,pos,neg]],columns=["Date","Compound Score","Pos Text","Neg Text"])
    df = df.append(row,ignore_index=True)
    return df

def sentiment_scores(text):
    # polarity_scores method of SentimentIntensityAnalyzer
    sentiment_dict = SentimentIntensityAnalyzer().polarity_scores(text)
    return(sentiment_dict['compound'])

def chatbot_responses(text,id):
    df = pd.read_csv(database,index_col="Index")
    compounddf = pd.read_csv(f"PFD2/Data_Folder/CompounScoreData{id}.csv",index_col="Index")
    check = df['ID'] == id
    name = df[df["ID"] == userID]["Name"].to_list()[0]
    #Updating Compoundf
    score = sentiment_scores(text)
    out = respond_sentiment(text,name) #Simulate Chatbot
    compounddf = adding_to_compound(compounddf,score,text)
    today = str(date.today())
    
    #Update database
    #fiveday = today - dt.timedelta(days=5)
    #tenday = today - dt.timedelta(days=10)
    compoundscore = sum(compounddf[compounddf["Date"] == today]["Compound Score"])/len(compounddf[compounddf["Date"] == today]["Compound Score"])
    #compoundscore5d = compounddf.loc[str(fiveday):str(today),"Compound Score"]
    #compoundscore10d = compounddf.loc[str(tenday):str(today),"Compound Score"]
    #df[check]["Compound Score"][0] = compoundscore
    print(compoundscore)
    df.loc[df.ID == id, "Compound Score"] = compoundscore
    print(df[check]["Compound Score"])
    compounddf.to_csv(f"PFD2/Data_Folder/CompounScoreData{id}.csv",index_label="Index")
    df.to_csv(database,index_label="Index")
    return out

def start_command(update, context):
    userID = None
    update.message.reply_text('Please "/Signup" or "/Login!"')

def help_command(update, context):
    update.message.reply_text("Here's the list of commands!\n/start\n/help\n/logout\n/login\n/signup\n/users\n/delete")

def signup_command(update,context):
    global signup
    signup = True
    update.message.reply_text("Please enter your Name (/cancel to cancel)")
    return NAME

def login_command(update,context):
    update.message.reply_text("Please enter your userID")
    return LOGINID

def logout_command(update,context):
    global userID
    userID = None
    update.message.reply_text("You have been Logged out!")

def users_command(update,context):
    df = pd.read_csv("PFD2/Data.csv",index_col="Index")
    lst = df[["ID","Name"]].values.tolist()
    out = [str(i)+". "+n for [i,n] in lst]
    out = ("\n").join(out)
    update.message.reply_text(out)

def del_command(update,context):
    df = pd.read_csv("PFD2/Data.csv",index_col="Index")
    lst = df[["ID","Name"]].values.tolist()
    out = [str(i)+". "+n for [i,n] in lst]
    out = ("\n").join(out)
    update.message.reply_text(f"{out}\nPlease enter ID to delete (/cancel to cancel):")
    return ID

def debug_command(update,context):
    global debugMode
    debugMode = True
    update.message.reply_text("Debug Mode Activated")

def handle_message(update, context):
    if userID != None:
        text = str(update.message.text).lower()
        response = chatbot_responses(text,userID)
    else:
        response = "Please do /signup or /login first!"
    if debugMode and userID != None:
        out = backendData(update.message.text)
        update.message.reply_text(out)
    else:
        sleep(3)
    update.message.reply_text(response)

def error(update, context):
    print(f"Update {update} caused error {context.error}")

def signup_name(update,context):
    global name
    name = update.message.text
    update.message.reply_text('Enter your age:')
    return AGE

def signup_age(update,context):
    global age
    age = update.message.text
    update.message.reply_text('Enter your profession:')
    return PROFESSION

def signup_profession(update,context):
    global profession
    profession = update.message.text
    update.message.reply_text('Enter the hours spent Studying/Working in a week:')
    return TIME

def signup_time(update,context):
    global timespent
    timespent = update.message.text
    update.message.reply_text('Enter the hours of exercise in a week:')
    return ACTIVENESS

def signup_activeness(update,context):
    global activeness
    activeness = update.message.text
    df = pd.read_csv(f"PFD2/Data.csv",index_col="Index")
    ID = df["ID"].max()+1
    tdy = date.today()
    row = pd.DataFrame([[ID,tdy,name,age,profession,timespent,activeness,0,0,0]],columns=["ID","LastUsed","Name","Age","Job/Profession","Time Spent Working/Studying (WK)","Activeness","Compound Score","5D Compound Score","10D Compound Score"])
    df = df.append(row,ignore_index=True)
    df.to_csv(f"PFD2/Data.csv",index_label="Index")
    df2 = pd.DataFrame(columns=["Date","Compound Score","Pos Text","Neg Text"])
    df2.to_csv(f"PFD2/Data_Folder/CompounScoreData{ID}.csv",index_label="Index")
    update.message.reply_text(f"Thank you for signing up!\nYour userID is: {ID}")
    return ConversationHandler.END

def cancel(update, context):
    """Cancels and ends the conversation."""
    update.message.reply_text('Operation Cancelled!')
    return ConversationHandler.END

def delete(update,context):
    global del_id
    del_id = int(update.message.text)
    df = pd.read_csv(f"PFD2/Data.csv",index_col="Index")
    idlist = df['ID'].to_list()
    if del_id in idlist:
        update.message.reply_text('Enter Admin Password(123):')
        return PASSWORD
    else:
        update.message.reply_text('ID not found try again!')
        return ID

def del_password(update,context):
    global del_pw
    del_pw = update.message.text
    if del_pw == "123":
        df = pd.read_csv(f"PFD2/Data.csv",index_col="Index")
        #Delete Data Row
        df = df[df["ID"] != del_id]
        lst = df[["ID","Name"]].values.tolist()
        out = [str(i)+". "+n for [i,n] in lst]
        out = ("\n").join(out)
        update.message.reply_text(out)
        update.message.reply_text("Successful!")
        #Delete Data File
        os.remove(f"PFD2/Data_Folder/CompounScoreData{del_id}.csv")
        df.to_csv(f"PFD2/Data.csv",index_label="Index")
        return ConversationHandler.END
    else:
        update.message.reply_text("Wrong Password! Try Again!")
        return PASSWORD

def login_id(update,context):
    global userID
    userID = int(update.message.text)
    df = pd.read_csv(database,index_col="Index")
    idList = df['ID'].to_list()
    if userID in idList:
        name = df[df["ID"] == userID]["Name"].to_list()[0]
        update.message.reply_text(f"Welcome Back {name}!")
        return ConversationHandler.END
    else:
        update.message.reply_text('ID not found try again!')
        return LOGINID

def main():
    updater = Updater(API_KEY, use_context=True)
    dp = updater.dispatcher

    signup_conv_handler = ConversationHandler(
    entry_points=[CommandHandler('Signup', signup_command)],
    states={
        NAME: [MessageHandler(Filters.text & ~Filters.command, signup_name)],
        AGE: [MessageHandler(Filters.regex(r'\d+'), signup_age)],
        PROFESSION: [MessageHandler(Filters.text & ~Filters.command, signup_profession)],
        TIME: [MessageHandler(Filters.text & ~Filters.command, signup_time)],
        ACTIVENESS: [MessageHandler(Filters.text & ~Filters.command, signup_activeness)],
        },
    fallbacks=[CommandHandler('cancel', cancel)])

    del_conv_handler = ConversationHandler(
    entry_points=[CommandHandler('delete', del_command)],
    states={
        ID: [MessageHandler(Filters.regex(r'\d+'), delete)],
        PASSWORD: [MessageHandler(Filters.regex(r'\d+'), del_password)],
        },
    fallbacks=[CommandHandler('cancel', cancel)])

    login_conv_handler = ConversationHandler(
    entry_points=[CommandHandler('Login', login_command)],
    states={
        LOGINID: [MessageHandler(Filters.regex(r'\d+'), login_id)]
        },
    fallbacks=[CommandHandler('cancel', cancel)])

    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("debug", debug_command))
    dp.add_handler(CommandHandler("logout", logout_command))
    dp.add_handler(del_conv_handler)
    dp.add_handler(signup_conv_handler)
    dp.add_handler(login_conv_handler)
    dp.add_handler(CommandHandler("users", users_command))
    dp.add_handler(MessageHandler(Filters.text, handle_message))


    dp.add_error_handler(error)

    #start_polling(20) = bot will wait 20 seconds before checking for the next user input
    updater.start_polling()

    #updater.idle allows bot to keep running
    updater.idle()

main()